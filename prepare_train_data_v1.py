import os, json, argparse, random
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
    for k in candidates:
        if k in ex and ex[k] not in (None, ""):
            try:
                return str(ex[k])
            except Exception:
                pass
    return str(idx)

def _merge(paths: List[str], shuffle: bool = True, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    merged: List[Dict[str, Any]] = []
    for p in paths:
        merged.extend(_read_jsonl(p))
    if shuffle:
        rng.shuffle(merged)
    return merged

def _split_train_dev(rows: List[Dict[str, Any]], dev_ratio: float, seed: int = 42):
    if dev_ratio <= 0.0 or len(rows) == 0:
        return rows, []
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    n_dev = max(1, int(len(rows) * dev_ratio))
    return rows[n_dev:], rows[:n_dev]

def _sample_fraction(rows: List[Dict[str, Any]], frac: float, seed: int = 42) -> List[Dict[str, Any]]:
    if frac <= 0.0 or len(rows) == 0:
        return []
    n = int(len(rows) * frac)
    rng = random.Random(seed)
    return rng.sample(rows, n) if n > 0 else []

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
        question = (
            f"Question: {question}\n\nChoices:\n" +
            "\n".join([f"{label}) {text}" for label, text in zip(choices.get("label", []), choices.get("text", []))])
        )
        gold = str(ex.get("answerKey", "")).strip()
        rows.append({
            "id": qid,
            "task_type": "arc-c",
            "question": question,
            "answerKey": gold,
            "answers": [gold] if gold else [],
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
            "task_type": "instruction_following",
            "instruction": instr,
            "context": ctx,
            "response": resp,
            "question": question,
            "answers": [resp] if resp else [],
        })

    _write_jsonl(out_path, rows)
    return out_path

def download_tulu_if_train(out_path: str) -> str:
    ds = load_dataset("allenai/tulu-3-sft-personas-instruction-following", split="train")
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        qid = _extract_id(ex, idx, candidates=["id"])
        prompt = str(ex.get("prompt", "") or "")
        messages = ex.get("messages", []) or []
        constraints = ex.get("constraints", []) or []

        user_parts = []
        assistant_resp = ""

        for m in messages:
            role = m.get("role", "")
            content = str(m.get("content", "") or "")
            if role == "assistant":
                assistant_resp = content
            else:
                tag = "User" if role in ("user","system") else role.capitalize()
                user_parts.append(f"{tag}: {content}")

        if user_parts:
            conversation_prefix = "\n".join(user_parts) + "\nAssistant:"
        else:
            conversation_prefix = f"User: {prompt}\nAssistant:"

        rows.append({
            "id": qid,
            "task_type": "instruction_following",
            "prompt": messages if messages else conversation_prefix,
            "constraints": constraints,
            "messages": messages,
            "question": conversation_prefix,
            "answers": [assistant_resp] if assistant_resp else [],
        })

    _write_jsonl(out_path, rows)
    return out_path

# ---------------- ARC oversampling ---------------- #
def oversample_arc_rows_with_shuffles(rows: List[Dict[str, Any]], shuffle_reps: int, seed: int) -> List[Dict[str, Any]]:
    if shuffle_reps <= 1:
        return rows
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(r)
        ch = r.get("choices", {}) or {}
        labels = ch.get("label", [])
        texts  = ch.get("text", [])
        gold   = (r.get("answerKey", "") or "").strip().upper()
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
                "answers": [new_gold],
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
    ap = argparse.ArgumentParser(description="Prepare separate training datasets + merged IT/SFT with dev split and optional replay.")
    ap.add_argument("--out-dir", default="data", help="Output directory")
    ap.add_argument("--triviaqa-subset", default="unfiltered.nocontext", help="HF subset for TriviaQA")
    ap.add_argument("--arc-mode", choices=["combined","separate"], default="combined")
    ap.add_argument("--arc-shuffle-reps", type=int, default=1, help="ARC oversampling (1=off).")
    ap.add_argument("--dev-ratio", type=float, default=0.02, help="Dev fraction on merged sets")
    ap.add_argument("--replay-frac", type=float, default=0.0, help="Fraction of IT to mix into SFT train (0=off)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(args.out_dir)

    triviaqa_path = os.path.join(out_dir, "triviaqa_train.jsonl")
    arc_combined_path = os.path.join(out_dir, "arc_e_c_train.jsonl")
    arc_aug_path      = os.path.join(out_dir, "arc_e_c_aug_train.jsonl")
    arc_c_path        = os.path.join(out_dir, "arc_c_train.jsonl")
    arc_e_path        = os.path.join(out_dir, "arc_e_train.jsonl")
    dolly_path        = os.path.join(out_dir, "dolly_train.jsonl")
    tulu_path         = os.path.join(out_dir, "tulu_train.jsonl")

    it_train_path         = os.path.join(out_dir, "it_train.jsonl")
    it_dev_path           = os.path.join(out_dir, "it_dev.jsonl")
    sft_tasks_train_path  = os.path.join(out_dir, "sft_train.jsonl")
    sft_tasks_dev_path    = os.path.join(out_dir, "sft_dev.jsonl")
    sft_tasks_replay_path = os.path.join(out_dir, "sft_replay_train.jsonl")

    print(f"Downloading TriviaQA (train) → {triviaqa_path}")
    download_triviaqa_train(triviaqa_path, subset=args.triviaqa_subset)

    print(f"Downloading ARC-Easy+Challenge (train) → {arc_combined_path}")
    download_arc_both_train(arc_combined_path)

    if args.arc_mode == "separate":
        print(f"Downloading ARC-Challenge (train) → {arc_c_path}")
        download_arc_c_train(arc_c_path)
        print(f"Downloading ARC-Easy (train) → {arc_e_path}")
        download_arc_e_train(arc_e_path)

    print(f"Downloading Dolly15K (train) → {dolly_path}")
    download_dolly_train(dolly_path)

    print(f"Downloading TULU (train) → {tulu_path}")
    download_tulu_if_train(tulu_path)

    # Optional ARC oversampling for the *combined* ARC file
    arc_to_use = arc_combined_path
    if args.arc_shuffle_reps > 1:
        print(f"Oversampling ARC-Easy+Challenge (train) → {arc_aug_path}")
        oversample_arc_file(arc_combined_path, arc_aug_path, args.arc_shuffle_reps, seed=args.seed)
        arc_to_use = arc_aug_path

    # -------- Stage A (IT) merge + dev split --------
    it_all = _merge([dolly_path, tulu_path], shuffle=True, seed=args.seed)
    it_train, it_dev = _split_train_dev(it_all, args.dev_ratio, seed=args.seed)
    _write_jsonl(it_train_path, it_train)
    _write_jsonl(it_dev_path, it_dev)

    # -------- Stage B (Tasks) merge + dev split --------
    sft_all = _merge([triviaqa_path, arc_to_use], shuffle=True, seed=args.seed)
    sft_train, sft_dev = _split_train_dev(sft_all, args.dev_ratio, seed=args.seed)

    # Optional replay: mix some IT into Stage-B train
    if args.replay_frac > 0.0:
        it_mix = _sample_fraction(it_train, args.replay_frac, seed=args.seed)
        sft_train = sft_train + it_mix
        random.Random(args.seed).shuffle(sft_train)
        _write_jsonl(sft_tasks_replay_path, sft_train)

    _write_jsonl(sft_tasks_train_path, sft_train)
    _write_jsonl(sft_tasks_dev_path, sft_dev)

    print("Done.")
    print(f"IT Train:          {len(it_train):6d}  → {it_train_path}")
    print(f"IT Dev:            {len(it_dev):6d}  → {it_dev_path}")
    print(f"SFT Train:         {len(sft_train):6d}  → {sft_tasks_train_path}")
    print(f"SFT Dev:           {len(sft_dev):6d}  → {sft_tasks_dev_path}")
    if args.replay_frac > 0.0:
        print(f"SFT Train (with replay) also written → {sft_tasks_replay_path}  (replay_frac={args.replay_frac:.2f})")

if __name__ == "__main__":
    main()
