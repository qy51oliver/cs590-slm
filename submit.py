#!/usr/bin/env python
import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Tuple

import torch
from zipfile import ZipFile, ZIP_DEFLATED

from pipelines import RouterPipeline, BasePipeline, OurPipeline
from score import score_from_rows
from generate import generate_rows, _write_jsonl, _read_jsonl



def set_random_seed(seed: int) -> None:

    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_numeric(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_numeric(v, key))
        else:
            try:
                num = float(v)
            except Exception:
                continue
            if math.isfinite(num):
                out[key] = num
    return out


def _aggregate_metrics(per_run_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    # Flatten each run's metrics, then compute mean/std for each numeric key
    flattened_runs: List[Dict[str, float]] = [_flatten_numeric(m) for m in per_run_metrics]
    all_keys: List[str] = sorted({k for m in flattened_runs for k in m.keys()})
    agg: Dict[str, Dict[str, float]] = {}
    for key in all_keys:
        vals: List[float] = [m[key] for m in flattened_runs if key in m]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        agg[key] = {"mean": mean, "std": std}
    return agg


def run_task(
    task: str,
    data_path: str,
    pipeline: BasePipeline,
    seeds: List[int],
    *,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    data_size: int,
) -> Tuple[List[Dict[str, Any]], List[List[str]], List[Dict[str, Any]]]:
    items = _read_jsonl(data_path)
    items = items[: data_size if data_size > 0 else None]

    per_run_preds: List[List[str]] = []
    per_run_metrics: List[Dict[str, Any]] = []

    for sd in seeds:
        set_random_seed(sd)
        rows = generate_rows(
            items,
            pipeline,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        preds = [str(r.get("prediction", "")) for r in rows]
        per_run_preds.append(preds)
        metrics = score_from_rows(task, rows)
        per_run_metrics.append(metrics)

    return items, per_run_preds, per_run_metrics


def main():
    ap = argparse.ArgumentParser(description="Prepare multi-run submissions and scores for Gradescope")
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--out_dir", default="submissions")
    ap.add_argument("--pipeline", default="router", choices=["router", "our", "base"])
    ap.add_argument("--data_triviaqa", default="data/triviaqa_test.jsonl")
    ap.add_argument("--data_arc_c", default="data/arc_c_test.jsonl")
    ap.add_argument("--data_ifeval", default="data/ifeval_test.jsonl")
    ap.add_argument("--data_size", type=int, default=1000, help="Number of items to process; -1 for all")

    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling for stochastic runs (recommended)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--seeds", type=str, default="1", help="Comma-separated seeds for repeated runs")

    args = ap.parse_args()

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    pipeline_map = {
        "router": RouterPipeline,
        "our": OurPipeline,
        "base": BasePipeline,
    }
    pipeline = pipeline_map[args.pipeline](args.model)

    task_to_path = {
        "triviaqa": args.data_triviaqa,
        "arc-c": args.data_arc_c,
        "ifeval": args.data_ifeval,
    }

    all_scores: Dict[str, Any] = {}
    written_jsonls: List[str] = []

    for task, path in task_to_path.items():
        if not os.path.exists(path):
            # Skip gracefully if a dataset is missing
            continue

        items, per_run_preds, per_run_metrics = run_task(
            task,
            path,
            pipeline,
            seeds,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(args.do_sample) or True,  # ensure stochasticity unless explicitly disabled by flag absence
            temperature=args.temperature,
            top_p=args.top_p,
            data_size=args.data_size,
        )

        # Build merged rows with predictions from all runs
        merged_rows: List[Dict[str, Any]] = []
        for idx, it in enumerate(items):
            row = {**it, "predictions": [run[idx] for run in per_run_preds]}
            rid = it.get("id")
            if rid is not None:
                row["id"] = rid
            merged_rows.append(row)

        out_path = os.path.join(args.out_dir, f"{task}_preds.jsonl")
        _write_jsonl(out_path, merged_rows)
        written_jsonls.append(out_path)

        # Aggregate metrics
        aggregate = _aggregate_metrics(per_run_metrics)
        all_scores[task] = {
            "n": len(items),
            "seeds": seeds,
            "per_run": per_run_metrics,
            "aggregate": aggregate,
        }

    # Write a combined scores file for convenience (store outside submissions dir)
    parent_dir = os.path.dirname(os.path.abspath(args.out_dir)) or "."
    scores_path = os.path.join(parent_dir, "scores.json")
    with open(scores_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    # Create zip archive with only the three task JSONLs
    zip_path = os.path.abspath(os.path.join(parent_dir, "submissions.zip"))
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        # Prefer canonical order and names
        expected = [
            os.path.join(args.out_dir, "triviaqa_preds.jsonl"),
            os.path.join(args.out_dir, "arc-c_preds.jsonl"),
            os.path.join(args.out_dir, "ifeval_preds.jsonl"),
        ]
        for fpath in expected:
            if os.path.exists(fpath):
                zf.write(fpath, arcname=os.path.basename(fpath))

    # Print concise summary
    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for task, data in all_scores.items():
        agg = data.get("aggregate", {})
        # Heuristic: pick a primary metric to display prominently
        primary_keys = [
            "acc",  # arc-c
            "avg_em_relax",  # triviaqa
            "eval_results_loose.instruction_accuracy",  # possible ifeval summary
        ]
        primary = None
        for k in primary_keys:
            if k in agg:
                primary = k
                break
        if primary is None and agg:
            primary = sorted(agg.keys())[0]
        if primary:
            m = agg[primary]
            summary[task] = {primary: (m.get("mean", 0.0), m.get("std", 0.0))}

    print(json.dumps({
        "out_dir": os.path.abspath(args.out_dir),
        "jsonls": [os.path.abspath(p) for p in written_jsonls],
        "zip": zip_path,
        "scores_json": scores_path,
        "summary": summary
    }, indent=2))


if __name__ == "__main__":
    main()


