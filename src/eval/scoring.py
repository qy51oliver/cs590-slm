#!/usr/bin/env python
import argparse
import json
import os
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


ARTICLES = {"a", "an", "the"}
PUNCT = set(string.punctuation)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows





def _normalize_text(t: str) -> str:
    t = t.strip().lower()
    t = " ".join(w for w in t.split() if w not in ARTICLES)
    t = "".join(ch for ch in t if ch not in PUNCT)
    return " ".join(t.split())


def _f1_em(pred: str, golds: List[str]) -> Tuple[float, int, int]:
    pred_n = _normalize_text(pred)
    if all(g is None or str(g).strip() == "" for g in golds):
        return (0.0, 0, 0)
    golds_n = [
        _normalize_text(g) for g in golds if g is not None and str(g).strip() != ""
    ]
    em = max(int(pred_n == g) for g in golds_n) if golds_n else 0
    em_relax = max(int(g in pred_n) for g in golds_n) if golds_n else 0

    def f1_one(g: str) -> float:
        pc, gc = Counter(pred_n.split()), Counter(g.split())
        overlap = sum((pc & gc).values())
        if overlap == 0:
            return 0.0
        prec = overlap / max(1, sum(pc.values()))
        rec = overlap / max(1, sum(gc.values()))
        return 2 * prec * rec / (prec + rec)

    f1 = max((f1_one(g) for g in golds_n), default=0.0)
    return f1, em, em_relax


def _get_triviaqa_golds(item: Dict[str, Any]) -> List[str]:
    # Support multiple common schemas
    if "answers" in item and isinstance(item["answers"], list):
        return [str(a) for a in item["answers"]]
    ans = item.get("answer") or {}
    if isinstance(ans, dict):
        if ans.get("normalized_aliases"):
            return list(ans["normalized_aliases"])  # type: ignore
        if ans.get("normalized_value"):
            return [str(ans["normalized_value"])]
        if ans.get("value"):
            return [str(ans["value"])]
    if "gold" in item:
        g = item["gold"]
        if isinstance(g, list):
            return [str(x) for x in g]
        return [str(g)]
    return []


def score_triviaqa(items: List[Dict[str, Any]], preds: List[str]) -> Dict[str, Any]:
    scores: List[Dict[str, Any]] = []
    for it, pred in zip(items, preds):
        golds = _get_triviaqa_golds(it)
        pred_line = str(pred).strip().split("\n")[0]
        f1, em, em_relax = _f1_em(pred_line, golds)
        scores.append({"f1": f1, "em": em, "em_relax": em_relax})
    n = len(scores)
    return {
        "task": "triviaqa",
        "n": n,
        "avg_f1": sum(s["f1"] for s in scores) / n if n else 0.0,
        "avg_em": sum(s["em"] for s in scores) / n if n else 0.0,
        "avg_em_relax": sum(s["em_relax"] for s in scores) / n if n else 0.0,
    }


def score_arc(items: List[Dict[str, Any]], preds: List[str]) -> Dict[str, Any]:
    correct = 0
    total = 0
    for it, pred in zip(items, preds):
        gold = str(it.get("answerKey", "")).strip().upper()
        pred_clean = str(pred).strip().upper()
        match = re.search(r"\b([A-E])\b", pred_clean)
        if match:
            letter = match.group(1)
        else:
            letter = pred_clean[0] if pred_clean and pred_clean[0].isalpha() else ""
        if gold and letter:
            correct += int(letter == gold)
            total += 1
    return {"task": "arc-c", "n": total, "acc": (correct / total) if total else 0.0}


def score_ifeval(items: List[Dict[str, Any]], preds: List[str]) -> Dict[str, Any]:
    # Prefer local package if available; fall back to adding path

    from ifeval.core.evaluation import InputExample as IFEvalInputExample
    from ifeval.languages.en.instructions import instruction_registry
    from ifeval.core.evaluation import Evaluator


    input_examples: List[IFEvalInputExample] = []
    for it in items:
        instr_ids = it.get("instruction_id_list") or []
        kwargs_list = it.get("kwargs") or [{}] * len(instr_ids)
        # remove the key that is none in kwargs
        kwargs_list = [{k: v for k, v in kwargs.items() if v is not None} for kwargs in kwargs_list]
        input_examples.append(
            IFEvalInputExample(
                instruction_id_list=instr_ids,
                prompt=it.get("prompt", ""),
                kwargs=kwargs_list,
            )
        )

    # Build mapping from prompt to prediction
    responses = {str(it.get("prompt", "")): str(pred) for it, pred in zip(items, preds)}

    evaluator = Evaluator(instruction_registry)
    report, _ = evaluator.evaluate(input_examples, responses)
    return {
        "task": "ifeval(pkg)",
        "n_prompts": len(input_examples),
        "eval_results_strict": report.get("eval_results_strict", {}),
        "eval_results_loose": report.get("eval_results_loose", {}),
    }


def score_from_rows(task: str, preds_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score directly from unified rows (item + "prediction").

    Args:
        task: One of "triviaqa", "arc-c", "ifeval"
        preds_rows: List of rows where each has the original item fields plus "prediction"

    Returns:
        Task-specific metrics dict
    """
    items = preds_rows
    preds: List[str] = [str(r.get("prediction", "")) for r in preds_rows]
    if task == "triviaqa":
        return score_triviaqa(items, preds)
    if task == "arc-c":
        return score_arc(items, preds)
    return score_ifeval(items, preds)


def main():
    ap = argparse.ArgumentParser(description="Score model outputs from a single unified JSONL or separate inputs/preds JSONLs")
    ap.add_argument("--task", default="triviaqa", choices=["triviaqa", "arc-c", "ifeval"])
    ap.add_argument("--preds", default="outputs/triviaqa_test_preds.jsonl", help="JSONL with predictions. Can also contain inputs+golds to act as a single unified file.")
    ap.add_argument("--out", default=None, help="Optional path to write metrics JSON")
    args = ap.parse_args()
    
    # args.task = "ifeval"
    # args.preds = "outputs/ifeval_test_preds.jsonl"
    preds_rows = _read_jsonl(args.preds)

    items = preds_rows
    preds = [r.get("prediction", "") for r in preds_rows]

    if args.task == "triviaqa":
        result = score_triviaqa(items, preds)
    elif args.task == "arc-c":
        result = score_arc(items, preds)
    else:
        result = score_ifeval(items, preds)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()



