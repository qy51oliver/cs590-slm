import argparse
import json
import os
from typing import Any, Dict, List

from pipelines import (
    RouterPipeline,
)
from generate import generate_rows, _write_jsonl, _read_jsonl
from score import score_from_rows


def __default_data_file_for_task(task: str) -> str:
    if task == "triviaqa":
        return "data/triviaqa_test.jsonl"
    if task == "arc-c":
        return "data/arc_c_test.jsonl"
    if task == "ifeval":
        return "data/ifeval_test.jsonl"
    raise ValueError(f"Unknown task: {task}")




def main():
    ap = argparse.ArgumentParser(description="Given a task, generate predictions and then score them from a single unified JSONL")
    ap.add_argument("--task", choices=["triviaqa", "arc-c", "ifeval"], default="arc-c")
    ap.add_argument("--data_file", default=None, help="Optional explicit data file; otherwise chosen by task")
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--out_dir", type=str, default="outputs", help="Directory to save predictions and metrics")
    ap.add_argument("--data-size", type=int, default=1000, help="Number of data items to process; -1 means all")

    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)

    args = ap.parse_args()
    data_file = args.data_file or __default_data_file_for_task(args.task)
    items = _read_jsonl(data_file)
    items = items[: args.data_size if args.data_size > 0 else None]

    pipeline = RouterPipeline(args.model)

    # args.temperature = 0.2

    # Route by task_type values produced by downloader: triviaqa, arc-c, ifeval
    rows = generate_rows(
        items,
        pipeline,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        )



    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.basename(data_file).replace(".jsonl", "")
    preds_path = os.path.join(args.out_dir, f"{base}_preds.jsonl")
    _write_jsonl(preds_path, rows)

    # Score using reusable scorer on the same unified rows
    result = score_from_rows(args.task, rows)

    metrics_path = os.path.join(args.out_dir, f"{base}_metrics.json")
    # with open(metrics_path, "w") as f:
    #     json.dump(result, f, indent=2)

    print(json.dumps({
        "task": args.task,
        "n": len(rows),
        "preds": os.path.abspath(preds_path),
        "metrics": os.path.abspath(metrics_path),
        **result,
    }, indent=2))


if __name__ == "__main__":
    main()



