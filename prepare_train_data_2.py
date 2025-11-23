import os, json, math, argparse, random
from typing import List, Dict, Any
from datasets import load_dataset

# ------------------------- utils -------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _write_jsonl(path: str, rows: List[Dict[str, Any]]):
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]
    
def _extract_id(ex: Dict[str, Any], idx: int, candidates: List[str]) -> str:
    """Return first available id-like field as string, else the index."""
    for k in candidates:
        if k in ex and ex[k] not in (None, ""):
            try:
                return str(ex[k])
            except Exception:
                pass
    return str(idx)

def _merge_and_write(paths: List[str], out_path: str, shuffle: bool = True, seed: int = 42) -> str:
    rng = random.Random(seed)
    merged: List[Dict[str, Any]] = []
    for p in paths:
        merged.extend(_read_jsonl(p))
    if shuffle:
        rng.shuffle(merged)
    _write_jsonl(out_path, merged)
    return out_path

# ---------- dataset builders ----------
def download_triviaqa_train(out_path: str, subset: str = "unfiltered.nocontext") -> str:
    ds = load_dataset("mandarjoshi/trivia_qa", subset, split="train")
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        qid = _extract_id(ex, idx, candidates=["id","qid","question_id","questionId","example_id","key"])
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

def arc_rows_from_subset(subset: str) -> List[Dict[str, Any]]:
    ds = load_dataset("allenai/ai2_arc", subset, split="train")
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        qid = _extract_id(ex, idx, ["id","qid","question_id","questionId","example_id","key"])
        choices = ex.get("choices", {}) or {}
        question = str(ex.get("question", ""))
        question = f"Question: {question}\n\nChoices:\n" + "\n".join([f"{label}) {text}" for label, text in zip(choices.get("label", []), choices.get("text", []))])
        rows.append({
            "id": qid,
            "task_type": "arc-c",
            "question": question,
            "answerKey": str(ex.get("answerKey", "")).strip(),
            "choices": choices,
        })
    return rows

def download_arc_both_train(out_path: str) -> str:
    rows = arc_rows_from_subset("ARC-Challenge") + arc_rows_from_subset("ARC-Easy")
    _write_jsonl(out_path, rows)
    return out_path

def download_arc_c_train(out_path: str) -> str:
    rows = arc_rows_from_subset("ARC-Challenge")
    _write_jsonl(out_path, rows)
    return out_path

def download_arc_e_train(out_path: str) -> str:
    rows = arc_rows_from_subset("ARC-Easy")
    _write_jsonl(out_path, rows)
    return out_path

def download_dolly_train(out_path: str) -> str:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        qid = _extract_id(ex, idx, candidates=["id", "instruction_id"])
        instr = str(ex.get("instruction", "") or "")
        ctx   = str(ex.get("context", "") or "")
        resp  = str(ex.get("response", "") or "")

        if ctx.strip():
            question = (
                "### Instruction:\n"
                f"{instr}\n\n"
                "### Input:\n"
                f"{ctx}\n\n"
                "### Response:\n"
            )
        else:
            question = (
                "### Instruction:\n"
                f"{instr}\n\n"
                "### Response:\n"
            )

        rows.append({
            "id": qid,
            "task_type": "dolly",
            "instruction": instr,
            "context": ctx,
            "response": resp,
            "question": question,
            "answers": [resp],
        })

    _write_jsonl(out_path, rows)
    return out_path



# ---------------- ARC oversampling ---------------- #
def oversample_arc_rows_with_shuffles(rows: List[Dict[str, Any]], shuffle_reps: int, seed: int) -> List[Dict[str, Any]]:
    """Label-preserving choice-order shuffles to increase ARC data."""
    if shuffle_reps <= 1:
        return rows
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(r)
        ch = r.get("choices", {}) or {}
        labels = ch.get("label", [])
        texts  = ch.get("text", [])
        gold   = r.get("answerKey", "").strip()
        if len(labels) != 4 or len(texts) != 4 or gold not in labels:
            continue
        stem = r["question"].split("\n\nChoices:\n")[0].replace("Question: ", "")
        gold_idx = labels.index(gold)
        for k in range(1, shuffle_reps):
            perm = list(range(4)); rng.shuffle(perm)
            new_texts = [texts[j] for j in perm]
            new_pos   = perm.index(gold_idx)
            new_gold  = labels[new_pos]
            question_fmt = "Question: " + stem + "\n\nChoices:\n" + "\n".join(
                f"{L}) {T}" for L, T in zip(labels, new_texts)
            )
            out.append({
                "id": f"{r['id']}-shuf{k}",
                "task_type": "arc-c",
                "question": question_fmt,
                "answerKey": new_gold,
                "choices": {"label": labels, "text": new_texts},
            })
    return out

def oversample_arc_file(in_path: str, out_path: str, shuffle_reps: int, seed: int = 42) -> str:
    rows = _read_jsonl(in_path)
    rows_aug = oversample_arc_rows_with_shuffles(rows, shuffle_reps, seed)
    _write_jsonl(out_path, rows_aug)
    return out_path    

# ---------------- main ---------------- #
def main():
    ap = argparse.ArgumentParser(description="Prepare separate training datasets + a merged training dataset.")
    ap.add_argument("--out-dir", default="data", help="Output directory")
    ap.add_argument("--triviaqa-subset", default="unfiltered.nocontext", help="HF subset for TriviaQA")
    ap.add_argument("--arc-mode", choices=["combined","separate"], default="combined", help="One file with Easy+Challenge (combined) or two files (separate).")
    ap.add_argument("--arc-shuffle-reps", type=int, default=1, help="ARC oversampling (1=off).")
    
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(args.out_dir)
    triviaqa_path = os.path.join(out_dir, "triviaqa_" + "train" + ".jsonl")
    arc_path = os.path.join(out_dir, "arc_e_c_" + "train" + ".jsonl")
    arc_aug_path = os.path.join(out_dir, "arc_e_c_aug_" + "train" + ".jsonl")
    dolly_path = os.path.join(out_dir, "dolly_train.jsonl")
    merged_path = os.path.join(out_dir, "sft_data.jsonl")
    
    print(f"Downloading TriviaQA (train) → {triviaqa_path}")
    download_triviaqa_train(triviaqa_path, subset="unfiltered.nocontext")

    print(f"Downloading ARC-Easy+Challenge (train) → {arc_path}")
    download_arc_both_train(arc_path)

    print(f"Downloading Dolly15K (train) → {dolly_path}")
    download_dolly_train(dolly_path)
    
    if args.arc_shuffle_reps > 1:
        print(f"Oversampling ARC-Easy+Challenge (train) → {arc_path}")
        oversample_arc_file(arc_path, arc_aug_path, args.arc_shuffle_reps)
        
    if args.arc_mode == "combined":
        print(f"Merging TriviaQA+ARC-Easy+Challenge → {merged_path}")
        arc_to_merge = arc_aug_path if args.arc_shuffle_reps > 1 else arc_path
        merged_path = _merge_and_write([triviaqa_path, arc_to_merge, dolly_path], merged_path, shuffle=True, seed=42)

    print("Done.")

if __name__ == "__main__":
    main()
    

    


    

