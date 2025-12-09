import os, json, random, argparse, hashlib
from datasets import load_dataset

LABELS = ["FQA", "REAS", "IF"]

# ---------------- helpers ----------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _write_jsonl(path, rows):
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} rows → {path}")

def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def _norm_q(x):
    return (x or "").strip()

def _hash(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _dedup_keep_first(rows):
    seen = set()
    out = []
    for r in rows:
        q = _norm_q(r.get("question"))
        if not q:
            continue
        h = _hash(q)
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out

def _print_counts(title, rows):
    c_fqa = sum(1 for r in rows if r["label"]=="FQA")
    c_rea = sum(1 for r in rows if r["label"]=="REAS")
    c_if  = sum(1 for r in rows if r["label"]=="IF")
    tot = len(rows)
    def pct(x): return 0.0 if tot==0 else 100.0*x/tot
    print(f"\n{title}")
    print(f"  FQA  : {c_fqa:7,d} ({pct(c_fqa):5.1f}%)")
    print(f"  REAS : {c_rea:7,d} ({pct(c_rea):5.1f}%)")
    print(f"  IF   : {c_if:7,d} ({pct(c_if):5.1f}%)")
    print(f"  TOTAL: {tot:7,d}")

# ---------------- loaders ----------------
def load_fqa(path, max_chars):
    rows = _read_jsonl(path)
    out = []
    for r in rows:
        q = _norm_q(r.get("question"))
        if not q or len(q) > max_chars:
            continue
        out.append({"id": f"fqa-{r.get('id','')}", "question": q, "label": "FQA"})
    return out

def load_reasoning(path, max_chars):
    rows = _read_jsonl(path)
    out = []
    for r in rows:
        q = _norm_q(r.get("question"))
        if not q or len(q) > max_chars:
            continue
        out.append({"id": f"reas-{r.get('id','')}", "question": q, "label": "REAS"})
    return out

def load_if(path, max_chars):
    rows = _read_jsonl(path)
    out = []
    for r in rows:
        q = _norm_q(r.get("question"))
        if not q or len(q) > max_chars:
            continue
        out.append({"id": f"if-{r.get('id','')}", "question": q, "label": "IF"})
    return out

# “hard IF” mining: code/math-y instructions (helps the router learn boundaries)
HARD_IF_PATTERNS = [
    "def ", "class ", "python", "leetcode", "write a function", "code", "compile",
    "sum of", "prime", "factor", "fibonacci", "big-o", "complexity",
    "integral", "derivative", "solve for", "equation", "matrix", "probability"
]
def mine_if_hard(if_rows):
    hard, easy = [], []
    for r in if_rows:
        q = r["question"].lower()
        is_hard = any(p in q for p in HARD_IF_PATTERNS)
        (hard if is_hard else easy).append(r)
    return hard, easy

def sample_k(xs, k, seed):
    rng = random.Random(seed)
    xs = list(xs)
    if len(xs) <= k:
        return xs
    idxs = rng.sample(range(len(xs)), k)
    return [xs[i] for i in idxs]

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Build 3-way router dataset (FQA/REAS/IF).")
    ap.add_argument("--fqa_file", default="data/v3/v3_factual_qa_train.jsonl")
    ap.add_argument("--reas_file", default="data/v3/v3_reasoning_train.jsonl")
    ap.add_argument("--if_file", default="data/v3/v3_instruction_following_train.jsonl")
    ap.add_argument("--out_dir", default="data/router_cls")

    ap.add_argument("--per_class", type=int, default=15000, help="target per class (FQA/REAS/IF)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.10)

    # length caps (pre-tokenization, plain chars)
    ap.add_argument("--fqa_max_chars", type=int, default=2000)
    ap.add_argument("--reas_max_chars", type=int, default=4000)
    ap.add_argument("--if_max_chars", type=int, default=3000)

    ap.add_argument("--if_hard_pct", type=float, default=0.20, help="fraction of IF from hard pool")

    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("Loading + per-class dedup…")
    fqa_all  = _dedup_keep_first(load_fqa(args.fqa_file,  args.fqa_max_chars))
    reas_all = _dedup_keep_first(load_reasoning(args.reas_file, args.reas_max_chars))
    if_all   = _dedup_keep_first(load_if(args.if_file, args.if_max_chars))

    _print_counts("Loaded (after per-class dedup, before sampling)", fqa_all+reas_all+if_all)

    print("\nMining hard IF (code/math-looking)…")
    if_hard, if_easy = mine_if_hard(if_all)
    print(f"  IF hard pool size: {len(if_hard):,} / IF total: {len(if_all):,}")

    print("\nBalanced sampling (FQA/REAS/IF)…")
    fqa = sample_k(fqa_all, args.per_class, args.seed)
    reas = sample_k(reas_all, args.per_class, args.seed+1)

    want_hard = int(round(args.per_class * max(0.0, min(1.0, args.if_hard_pct))))
    want_easy = args.per_class - want_hard
    if_h = sample_k(if_hard, want_hard, args.seed+2)
    if_e = sample_k(if_easy, want_easy, args.seed+3)
    iff = if_h + if_e

    _print_counts("After balanced sampling (FQA/REAS/IF)", fqa+reas+iff)

    print("\nMerging + global dedup + shuffle…")
    merged = fqa + reas + iff
    before = len(merged)
    merged = _dedup_keep_first(merged)
    removed = before - len(merged)
    if removed > 0:
        print(f"  Removed {removed:,} near-duplicates across classes.")
    rng.shuffle(merged)

    _print_counts("Final merged set (pre-split)", merged)

    val_frac = max(0.0, min(0.5, float(args.val_frac)))
    _ensure_dir(args.out_dir)
    train_path = os.path.join(args.out_dir, "router_cls_train.jsonl")

    if 0.0 < val_frac < 0.5 and len(merged) >= 1000:
        cut = int(len(merged) * (1.0 - val_frac))
        train_rows = merged[:cut]
        val_rows = merged[cut:]
        _write_jsonl(train_path, train_rows)
        _write_jsonl(os.path.join(args.out_dir, "router_cls_val.jsonl"), val_rows)
    else:
        _write_jsonl(train_path, merged)

    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS,
                   "desc": {"FQA":"factual_qa", "REAS":"reasoning", "IF":"instruction_following"}}, f, indent=2)
    print("\nDone.")

if __name__ == "__main__":
    main()