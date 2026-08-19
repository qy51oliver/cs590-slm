#!/usr/bin/env python
import os
import json
import argparse
from typing import List, Dict, Any

from datasets import load_dataset


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]):
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_id(ex: Dict[str, Any], idx: int, candidates: List[str]) -> str:
    """Return first available id-like field as string, else the index."""
    for key in candidates:
        if key in ex and ex[key] not in (None, ""):
            try:
                return str(ex[key])
            except Exception:
                pass
    return str(idx)


def download_triviaqa(out_path: str, subset: str = "unfiltered.nocontext", split: str = "test") -> str:
    """Download TriviaQA and export to JSONL with keys aligned to scripts/score.py.

    Output schema per line:
      - question: str
      - answers: List[str]   # score.py reads this via _get_triviaqa_golds
    """
    # TriviaQA test set does not have gold answers; use validation instead
    if split == "test":
        split = "validation"
    ds = load_dataset("mandarjoshi/trivia_qa", subset, split=split)
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        qid = _extract_id(ex, idx, candidates=[
            "id", "qid", "question_id", "questionId", "example_id", "key"
        ])
        q = ex.get("question", "")
        ans = ex.get("answer", {}) or {}
        aliases = ans.get("normalized_aliases") or []
        main_val = ans.get("normalized_value") or ans.get("value")
        golds: List[str] = [str(a) for a in aliases] if aliases else ([str(main_val)] if main_val else [])
        rows.append({
            "id": qid,
            "task_type": "triviaqa",
            "question": str(q),
            "answers": golds,
        })
    _write_jsonl(out_path, rows)
    return out_path


def download_arc_c(out_path: str, split: str = "test") -> str:
    """Download ARC-Challenge (ai2_arc) and export to JSONL with keys aligned to scripts/score.py.

    Output schema per line:
      - question: str
      - answerKey: str       # required by score_arc
      - choices: {label: List[str], text: List[str]} (optional but useful)
    """
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        qid = _extract_id(ex, idx, candidates=[
            "id", "qid", "question_id", "questionId", "example_id", "key"
        ])
        # add question to choices
        choices = ex.get("choices", {})
        question = str(ex.get("question", ""))
        question = f"Question: {question}\n\nChoices:\n" + "\n".join([f"{label}) {text}" for label, text in zip(choices.get("label", []), choices.get("text", []))])
        
        rows.append({
            "id": qid,
            "task_type": "arc-c",
            "question": question,
            "answerKey": str(ex.get("answerKey", "")),
            "choices": ex.get("choices", {}),
        })
    _write_jsonl(out_path, rows)
    return out_path


def download_ifeval(out_path: str, split: str = "test") -> str:
    """Download google/IFEval and export to JSONL with keys aligned to scripts/score.py.

    Output schema per line:
      - question: str                     # unified key (duplicate of prompt)
      - prompt: str                       # required by score_ifeval
      - instruction_id_list: List[str]
      - kwargs: List[Dict[str, Any]]      # same length as instruction_id_list
    """
    split = "train" if split == "test" else split  # IFEval uses 'train' as test split
    ds = load_dataset("google/IFEval", split=split)
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        qid = _extract_id(ex, idx, candidates=[
            "key", "id", "qid", "question_id", "questionId", "example_id"
        ])
        prompt = str(ex.get("prompt", ""))
        rows.append({
            "id": qid,
            "task_type": "ifeval",
            "question": prompt,
        } | ex)
        # instr_ids = ex.get("instruction_id_list") or []
        # kwargs_list = ex.get("kwargs") or [{}] * len(instr_ids)
        # # Ensure lengths match for downstream scorers
        # if isinstance(kwargs_list, list) and len(kwargs_list) != len(instr_ids):
        #     # Pad or trim to match instruction ids length
        #     if len(kwargs_list) < len(instr_ids):
        #         kwargs_list = kwargs_list + [{}] * (len(instr_ids) - len(kwargs_list))
        #     else:
        #         kwargs_list = kwargs_list[:len(instr_ids)]
        # rows.append({
        #     "id": qid,
        #     "task_type": "ifeval",
        #     "question": prompt,
        #     "prompt": prompt,
        #     "instruction_id_list": list(instr_ids),
        #     "kwargs": list(kwargs_list),
        # })
    _write_jsonl(out_path, rows)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Download and normalize eval datasets to JSONL in ./data")
    ap.add_argument("--out-dir", default="data", help="Output directory for JSONL files")
    ap.add_argument("--triviaqa-subset", default="unfiltered.nocontext", help="HF subset for TriviaQA")
    ap.add_argument("--triviaqa-split", default="test", help="Split for TriviaQA (train/validation/test)")
    ap.add_argument("--arc-split", default="test", help="Split for ARC-Challenge (train/validation/test)")
    ap.add_argument("--ifeval-split", default="test", help="Split for google/IFEval (train as test)")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(out_dir)

    triviaqa_path = os.path.join(out_dir, "triviaqa_" + args.triviaqa_split + ".jsonl")
    arc_path = os.path.join(out_dir, "arc_c_" + args.arc_split + ".jsonl")
    ifeval_path = os.path.join(out_dir, "ifeval_" + args.ifeval_split + ".jsonl")

    print(f"Downloading TriviaQA ({args.triviaqa_subset}, split={args.triviaqa_split}) → {triviaqa_path}")
    download_triviaqa(triviaqa_path, subset=args.triviaqa_subset, split=args.triviaqa_split)

    print(f"Downloading ARC-Challenge (split={args.arc_split}) → {arc_path}")
    download_arc_c(arc_path, split=args.arc_split)

    print(f"Downloading IFEval (split={args.ifeval_split}) → {ifeval_path}")
    download_ifeval(ifeval_path, split=args.ifeval_split)

    print("Done.")


if __name__ == "__main__":
    main()


