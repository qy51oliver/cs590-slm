#!/usr/bin/env python3
import argparse
import json
import gc
import os
from typing import Any, Dict, List

import torch
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from src.inference.pipeline import OurPipeline
from src.inference.generation import generate_rows, _read_jsonl
from src.eval.scoring import score_from_rows


# ------------------------- helpers -------------------------
def default_data_file_for_task(task: str, data_dir: str) -> str:
    """
    Fallback data path if per-task overrides are not provided.
    """
    if task == "triviaqa":
        return os.path.join(data_dir, "triviaqa_test.jsonl")
    if task == "arc-c":
        return os.path.join(data_dir, "arc_c_test.jsonl")
    if task == "ifeval":
        return os.path.join(data_dir, "ifeval_test.jsonl")
    raise ValueError(f"Unknown task: {task}")


def extract_scalar_metric(task: str, metrics: Dict[str, Any]) -> float:
    """
    Extract the main scalar metric for each task:

    - TriviaQA: avg_em_relax
    - ARC-C: acc
    - IFEval: eval_results_loose.instruction_accuracy
    """
    if task == "triviaqa":
        return float(metrics.get("avg_em_relax", 0.0))
    if task == "arc-c":
        return float(metrics.get("acc", 0.0))
    if task == "ifeval":
        loose = metrics.get("eval_results_loose", {}) or {}
        return float(loose.get("instruction_accuracy", 0.0))
    raise ValueError(f"Unknown task for scalar metric: {task}")


def compute_global_score(cfg: Dict[str, Any], tasks: List[str]) -> float:
    """
    Global score = sum of per-task scalar metrics.
    """
    return sum(cfg["tasks"][t]["scalar"] for t in tasks)


def resolve_data_path(task: str, args) -> str:
    """
    优先使用 per-task override，其次使用 data_dir 默认路径。
    """
    if task == "triviaqa" and args.data_triviaqa:
        return args.data_triviaqa
    if task == "arc-c" and args.data_arc_c:
        return args.data_arc_c
    if task == "ifeval" and args.data_ifeval:
        return args.data_ifeval
    return default_data_file_for_task(task, args.data_dir)


def to_jsonable(obj):
    """
    Recursively convert numpy types to Python native types so json.dump won't crash.
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(to_jsonable(x) for x in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Bayesian Optimization ablation for TriviaQA / ARC-C / IFEval"
    )

    # Tasks
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["triviaqa", "arc-c", "ifeval"],
        choices=["triviaqa", "arc-c", "ifeval"],
        help="Tasks to include in the ablation search.",
    )

    # Data config
    # 支持 --data-dir 和 --data_dir 两种写法，统一映射到 args.data_dir
    ap.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        type=str,
        default="data",
        help="Base directory containing *_test.jsonl files.",
    )
    ap.add_argument(
        "--data_size",
        type=int,
        default=1000,
        help="Number of examples per task (-1 = use all).",
    )

    # Per-task data overrides
    ap.add_argument(
        "--data_triviaqa",
        type=str,
        default=None,
        help="Optional explicit data file for TriviaQA.",
    )
    ap.add_argument(
        "--data_arc_c",
        type=str,
        default=None,
        help="Optional explicit data file for ARC-C.",
    )
    ap.add_argument(
        "--data_ifeval",
        type=str,
        default=None,
        help="Optional explicit data file for IFEval.",
    )

    # Batch size
    ap.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for generate_rows().",
    )

    # Model paths / router config
    ap.add_argument(
        "--router_model",
        type=str,
        default="oliveryql/gemma270m-sft-router",
    )
    ap.add_argument(
        "--fqa_model",
        type=str,
        default="oliveryql/gemma270m-sft-fqa",
    )
    ap.add_argument(
        "--reas_model",
        type=str,
        default="oliveryql/gemma270m-sft-reasoning",
    )
    ap.add_argument(
        "--if_model",
        type=str,
        default="oliveryql/gemma270m-sft-if",
    )
    ap.add_argument(
        "--router_max_len",
        type=int,
        default=1024,
    )
    ap.add_argument(
        "--evict-after-route",
        dest="evict_after_route",
        action="store_true",
        help="Evict expert models from GPU after each routing bucket to save memory.",
    )

    # Bayesian Optimization settings
    ap.add_argument(
        "--n_calls",
        type=int,
        default=25,
        help="Number of BO evaluations.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="ablation_bo_results.json",
        help="Path where incremental BO results (JSON) will be stored.",
    )

    args = ap.parse_args()
    tasks = args.tasks

    # ---------------- Load datasets ----------------
    print("[INFO] Loading datasets...")
    task_items: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        data_file = resolve_data_path(task, args)
        items = _read_jsonl(data_file)
        if args.data_size > 0:
            items = items[: args.data_size]
        task_items[task] = items
        print(f"[INFO] Loaded {len(items)} examples for task '{task}' from {data_file}")

    # ---------------- Search space ----------------
    # 搜索 4 个参数：
    # - reasoning_sc_K: self-consistency 投票次数
    # - max_new_tokens: 最大生成长度
    # - temperature, top_p: 采样超参数
    space = [
        Integer(1, 15, name="reasoning_sc_K"),
        Integer(256, 1024, name="max_new_tokens"),
        Real(0.3, 1.3, name="temperature"),
        Real(0.6, 1.0, name="top_p"),
    ]

    # ---------------- Objective function ----------------
    @use_named_args(space)
    def objective(**params):
        print("\n==================================================")
        print("[BO] Evaluating params:", params)
        print("==================================================")

        # Build router pipeline with given K
        pipeline = OurPipeline(
            router_model=args.router_model,
            experts={
                "factual_qa": args.fqa_model,
                "reasoning": args.reas_model,
                "instruction_following": args.if_model,
            },
            router_max_len=args.router_max_len,
            evict_after_route=args.evict_after_route,
            reasoning_sc_K=params["reasoning_sc_K"],
        )

        cfg: Dict[str, Any] = {
            "reasoning_sc_K": params["reasoning_sc_K"],
            "max_new_tokens": params["max_new_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "tasks": {},
        }

        # Evaluate each task
        for task in tasks:
            rows = generate_rows(
                task_items[task],
                pipeline,
                batch_size=args.batch_size,
                max_new_tokens=params["max_new_tokens"],
                do_sample=True,  # 这里统一启用采样，把 temperature / top_p 纳入搜索
                temperature=params["temperature"],
                top_p=params["top_p"],
            )
            metrics = score_from_rows(task, rows)
            scalar = extract_scalar_metric(task, metrics)
            cfg["tasks"][task] = {
                "scalar": scalar,
                "metrics": metrics,
            }
            print(f"[RESULT] {task}: scalar={scalar:.4f}")

        # Cleanup
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        global_score = compute_global_score(cfg, tasks)
        cfg["global_score"] = global_score
        print(f"[GLOBAL SCORE] {global_score:.4f}\n")

        # Save results incrementally (JSON-safe)
        try:
            with open(args.out, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

        history.append(to_jsonable(cfg))

        with open(args.out, "w") as f:
            json.dump(history, f, indent=2)

        # skopt 是最小化目标函数，这里取负号实现“最大化 global_score”
        return -global_score

    # ---------------- Run Bayesian Optimization ----------------
    print("\n[INFO] Starting Bayesian Optimization...\n")
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=args.n_calls,
        random_state=42,
        n_initial_points=5,
    )

    print("\n================ BEST CONFIG FOUND ================\n")
    print("Best parameters (reasoning_sc_K, max_new_tokens, temperature, top_p):")
    print(result.x)
    print("Best approx global score:", -result.fun)
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
