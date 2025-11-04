#!/usr/bin/env python
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Callable

from pipelines import (
    RouterPipeline,
    BasePipeline,
)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    




def generate_rows(
    items: List[Dict[str, Any]],
    pipeline: BasePipeline,
    **gen_kwargs,
) -> List[Dict[str, Any]]:
    """Generate predictions for items using a provided pipeline runner and
    return unified rows (original item merged with {"prediction": str}).

    Args:
        items: Task-specific input items
        pipeline_run: Callable that accepts (items, **gen_kwargs) and returns List[str] predictions
        **gen_kwargs: Generation kwargs forwarded to the pipeline runner

    Returns:
        List of rows with the original item fields plus a "prediction" key
    """
    preds = pipeline.run(items, **gen_kwargs)
    rows: List[Dict[str, Any]] = []
    for it, pred in zip(items, preds):
        rid = it.get("id")
        row = {"prediction": pred} | it
        if rid is not None:
            row["id"] = rid
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Generation pipeline: read queries JSONL, output predictions JSONL")

    ap.add_argument("--data_file", default="data/triviaqa_test.jsonl",
                    help="Input JSONL with queries (task-specific schema)")
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--out_dir", type=str, default="outputs", help="Output JSONL with predictions aligned to inputs")
    ap.add_argument("--data-size", type=int, default=500, help="Number of data items to process; -1 means all")

    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    


    args = ap.parse_args()
    args.data_file = "data/arc_c_test.jsonl"
    items = _read_jsonl(args.data_file)
    items = items[: args.data_size if args.data_size > 0 else None]

    pipeline = RouterPipeline(args.model)


    rows = generate_rows(
        items,
        pipeline,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    out_path = os.path.join(args.out_dir,
                            os.path.basename(args.data_file).replace(".jsonl", f"_preds.jsonl"))
    _write_jsonl(out_path, rows)

    print({"n": len(rows), "out": os.path.abspath(out_path)})
    return rows


if __name__ == "__main__":
    main()



