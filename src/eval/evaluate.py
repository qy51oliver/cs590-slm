import argparse
import json
import os
from typing import Any, Dict, List

from src.inference.pipeline import (
    RouterPipeline,
    OurPipeline,
)
from src.inference.generation import generate_rows, _write_jsonl, _read_jsonl
from src.eval.scoring import score_from_rows


def __default_data_file_for_task(task: str) -> str:
    if task == "triviaqa":
        return "data/triviaqa_test.jsonl"
    if task == "arc-c":
        return "data/arc_c_test.jsonl"
    if task == "ifeval":
        return "data/ifeval_test.jsonl"
    raise ValueError(f"Unknown task: {task}")


def main():
    ap = argparse.ArgumentParser(
        description="Given a task, generate predictions and then score them from a single unified JSONL"
    )
    ap.add_argument("--task", choices=["triviaqa", "arc-c", "ifeval"], default="arc-c")
    ap.add_argument("--data_file", default=None, help="Optional explicit data file; otherwise chosen by task")
    # ap.add_argument("--model", default="google/gemma-3-270m")
    ap.add_argument("--out_dir", type=str, default="outputs", help="Directory to save predictions and metrics")
    ap.add_argument("--data-size", type=int, default=1000, help="Number of data items to process; -1 means all")

    # ===== Best config defaults: K=5, T=763, temp=0.43894, top_p=0.6 =====
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=763)
    ap.add_argument(
        "--do-sample",
        action="store_true",
        help="If set, enable sampling for non-reasoning routes (reasoning still uses SC sampling).",
    )
    ap.add_argument("--temperature", type=float, default=0.43894076403615506)
    ap.add_argument("--top-p", type=float, default=0.6)

    # Router + experts config
    ap.add_argument("--router_model", type=str, default="oliveryql/gemma270m-sft-router")
    ap.add_argument("--fqa_model", type=str, default="oliveryql/gemma270m-sft-fqa")
    ap.add_argument("--reas_model", type=str, default="oliveryql/gemma270m-sft-reasoning")
    ap.add_argument("--if_model", type=str, default="oliveryql/gemma270m-sft-if")
    ap.add_argument("--router_max_len", type=int, default=1024)
    ap.add_argument(
        "--evict_after_route",
        action="store_true",
        help="Unload each expert after finishing its bucket to reduce VRAM.",
    )

    # OurPipeline-specific hyperparameters
    ap.add_argument(
        "--reasoning_sc_K",
        type=int,
        default=5,
        help="Self-consistency K for reasoning route (ARC-C). Best config uses K=5.",
    )
    ap.add_argument(
        "--factual_qa_max_new_tokens",
        type=int,
        default=64,
        help="Max new tokens for the first-stage QA draft. Default 64.",
    )

    args = ap.parse_args()
    data_file = args.data_file or __default_data_file_for_task(args.task)
    items = _read_jsonl(data_file)
    items = items[: args.data_size if args.data_size > 0 else None]

    # OurPipeline with best defaults (and CLI overrides)
    pipeline = OurPipeline(
        router_model=args.router_model,
        experts={
            "factual_qa": args.fqa_model,
            "reasoning": args.reas_model,
            "instruction_following": args.if_model,
        },
        router_max_len=args.router_max_len,
        evict_after_route=args.evict_after_route,
        reasoning_sc_K=args.reasoning_sc_K,
    )

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

    print(
        json.dumps(
            {
                "task": args.task,
                "n": len(rows),
                "preds": os.path.abspath(preds_path),
                "metrics": os.path.abspath(metrics_path),
                "reasoning_sc_K": args.reasoning_sc_K,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                **result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
