import argparse
import json
import os
from typing import Any, Dict, List
import re

from pipelines_qa_strategies import BaseFQAPipeline


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} rows → {path}")


# ---------------- TriviaQA metrics (EM / F1) ----------------
def _normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # remove English articles
    articles = {"a", "an", "the"}
    tokens = [t for t in s.split() if t not in articles]
    return " ".join(tokens)


def _f1_single(pred: str, gold: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = 0
    gold_counts: Dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    for t in pred_tokens:
        if gold_counts.get(t, 0) > 0:
            common += 1
            gold_counts[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _score_triviaqa_row(pred: str, gold_answers: List[str]) -> Dict[str, float]:
    if not gold_answers:
        gold_answers = [""]
    em = 0.0
    best_f1 = 0.0
    norm_pred = _normalize_answer(pred)
    for g in gold_answers:
        norm_g = _normalize_answer(g)
        if norm_pred == norm_g and norm_g != "":
            em = 1.0
        f1 = _f1_single(pred, g)
        if f1 > best_f1:
            best_f1 = f1
    return {"em": em, "f1": best_f1}


def eval_triviaqa(
    data: List[Dict[str, Any]],
    preds: List[str],
) -> Dict[str, float]:
    assert len(data) == len(preds)
    total_em = 0.0
    total_f1 = 0.0
    n = len(data)
    for ex, p in zip(data, preds):
        gold = ex.get("answers") or ex.get("answer") or []
        if isinstance(gold, str):
            gold = [gold]
        scores = _score_triviaqa_row(p, gold)
        total_em += scores["em"]
        total_f1 += scores["f1"]
    avg_em = total_em / n if n > 0 else 0.0
    avg_f1 = total_f1 / n if n > 0 else 0.0
    return {"avg_em": avg_em, "avg_f1": avg_f1, "n": n}


def main():
    ap = argparse.ArgumentParser(description="Evaluate multiple QA inference strategies on TriviaQA.")
    ap.add_argument("--model", type=str, default="oliveryql/gemma270m-sft-fqa")
    ap.add_argument("--data_file", type=str, default="data/triviaqa_test.jsonl")
    ap.add_argument("--out_dir", type=str, default="outputs/qa_strategies")
    ap.add_argument("--n", type=int, default=1000, help="Number of examples to evaluate (0 = all).")
    ap.add_argument(
        "--strategies",
        type=str,
        default="greedy_64,greedy_128,refine_32_16,refine_64_16,combo_greedy_refine_if_long,sample3",
        help="Comma-separated list of strategies to evaluate.",
    )
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    print(f"Loading TriviaQA data from: {args.data_file}")
    data = _read_jsonl(args.data_file)
    if args.n > 0:
        data = data[: args.n]
    print(f"Loaded {len(data):,} examples")

    pipe = BaseFQAPipeline(model_name=args.model)

    _ensure_dir(args.out_dir)
    summary: Dict[str, Any] = {
        "task": "triviaqa",
        "data_file": args.data_file,
        "model": args.model,
        "n": len(data),
        "results": {},
    }

    for strat in strategies:
        print(f"\n=== Strategy: {strat} ===")
        preds = pipe.run_strategy(data, strategy=strat, batch_size=args.batch_size)
        rows_with_pred: List[Dict[str, Any]] = []
        for ex, p in zip(data, preds):
            r = dict(ex)
            r["prediction"] = p
            rows_with_pred.append(r)
        pred_path = os.path.join(args.out_dir, f"triviaqa_{strat}_preds.jsonl")
        _write_jsonl(pred_path, rows_with_pred)

        metrics = eval_triviaqa(data, preds)
        metrics["preds_file"] = pred_path
        summary["results"][strat] = metrics

        print(
            f"Strategy {strat}: n={metrics['n']}, "
            f"avg_em={metrics['avg_em']:.4f}, avg_f1={metrics['avg_f1']:.4f}"
        )

    summary_path = os.path.join(args.out_dir, "triviaqa_eval_strategies.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote summary → {summary_path}")


if __name__ == "__main__":
    main()
