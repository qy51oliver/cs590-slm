import os, json, random, argparse, hashlib, re
from datasets import load_dataset

LABELS = ["FQA", "REAS", "IF", "OTHER"]

# --------- utils ---------

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

_norm_space = re.compile(r"\s+")
def _normalize_question(s):
    s = (s or "").strip()
    s = _norm_space.sub(" ", s)
    return s.lower()

def _hash_norm(s):
    return hashlib.md5(_normalize_question(s).encode("utf-8")).hexdigest()

def _dedup_keep_first(rows):
    seen = set()
    out = []
    for r in rows:
        q = (r.get("question") or "").strip()
        if not q:
            continue
        h = _hash_norm(q)
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out

def _report_counts(tag, rows):
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    n = len(rows)
    print(f"\n{tag}")
    for lab in LABELS:
        v = c.get(lab, 0)
        pct = (100.0 * v / n) if n else 0.0
        print(f"  {lab:<5}: {v:7,d} ({pct:5.1f}%)")
    print(f"  TOTAL: {n:,}")

# --------- IF heuristics (for mining "hard IF" positives) ---------

_CODE_MATH_IF = re.compile(
    r"```|(?:^|\b)(write|implement|complete)\s+(?:a|an)?\s*(function|program|class)\b"
    r"|def\s+\w+\s*\(|class\s+\w+|import\s+\w+|time\s+complexity|big-?o"
    r"|prove\b|show\s+that\b|derive\b|\\sum|\\frac|\\int",
    re.IGNORECASE
)

def looks_like_if_code_math(text):
    return bool(_CODE_MATH_IF.search(text or ""))

# --------- loaders for 3 experts ---------

def load_fqa(path, max_chars=2000):
    rows = _read_jsonl(path)
    out = []
    for r in rows:
        q = (r.get("question") or "").strip()
        if not q or len(q) > max_chars:
            continue
        out.append({"id": f"fqa-{r.get('id','')}", "question": q, "label": "FQA"})
    return _dedup_keep_first(out)

def load_reasoning(path, max_chars=4000):
    rows = _read_jsonl(path)
    out = []
    for r in rows:
        q = (r.get("question") or "").strip()
        if not q or len(q) > max_chars:
            continue
        out.append({"id": f"reas-{r.get('id','')}", "question": q, "label": "REAS"})
    return _dedup_keep_first(out)

def load_if(path, max_chars=4000):
    rows = _read_jsonl(path)
    out = []
    for r in rows:
        q = (r.get("question") or "").strip()
        if not q or len(q) > max_chars:
            continue
        out.append({"id": f"if-{r.get('id','')}", "question": q, "label": "IF"})
    return _dedup_keep_first(out)

# --------- OTHER samplers: gsm8k + mbpp only ---------

def load_gsm8k(n, seed, max_chars=1200):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), k=min(n, len(ds)))
    out = []
    for i in idxs:
        q = (ds[i].get("question") or "").strip()
        if not q:
            continue
        if len(q) > max_chars:
            q = q[:max_chars]
        out.append({"id": f"gsm8k-{i}", "question": q, "label": "OTHER"})
    return _dedup_keep_first(out)

def load_mbpp(n, seed, max_chars=800):
    try:
        ds = load_dataset("mbpp", split="train")
    except Exception:
        ds = load_dataset("google-research-datasets/mbpp", split="train")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), k=min(n, len(ds)))
    out = []
    for i in idxs:
        prompt = ds[i].get("text") or ds[i].get("prompt") or ds[i].get("description") or ""
        prompt = str(prompt).strip()
        if not prompt:
            continue
        q = f"Write a Python function:\n{prompt}"
        if len(q) > max_chars:
            q = q[:max_chars]
        out.append({"id": f"mbpp-{i}", "question": q, "label": "OTHER"})
    return _dedup_keep_first(out)

def build_other(total, seed):
    # Split OTHER between GSM8K and MBPP; if small corpora, sampling handles the cap.
    n_g = total // 2
    n_m = total - n_g
    out = []
    out += load_gsm8k(n_g, seed+10)
    out += load_mbpp(n_m, seed+11)
    # Dedup across OTHER
    out = _dedup_keep_first(out)
    return out

# --------- sampling helpers ---------

def sample_balanced(xs, k, seed):
    rng = random.Random(seed)
    xs = _dedup_keep_first(xs)
    if len(xs) <= k:
        return xs
    idxs = rng.sample(range(len(xs)), k=k)
    return [xs[i] for i in idxs]

def mine_if_hard_cases(rows, max_chars):
    hard = []
    for r in rows:
        q = (r.get("question") or "").strip()
        if not q or len(q) > max_chars:
            continue
        if looks_like_if_code_math(q):
            hard.append(r)
    return hard

# --------- main ---------

def main():
    ap = argparse.ArgumentParser(description="Build router classification dataset (FQA/REAS/IF/OTHER).")
    # defaults matched to your repo layout
    ap.add_argument("--fqa_file", default="data/v3/v3_factual_qa_train.jsonl")
    ap.add_argument("--reas_file", default="data/v3/v3_reasoning_train.jsonl")
    ap.add_argument("--if_file", default="data/v3/v3_instruction_following_train.jsonl")
    ap.add_argument("--out_dir", default="data/router_cls")

    ap.add_argument("--per_class", type=int, default=15000, help="examples per class for FQA/REAS/IF")
    ap.add_argument("--other_total", type=int, default=12000, help="total OTHER examples")

    ap.add_argument("--fqa_max_chars", type=int, default=2000)
    ap.add_argument("--reas_max_chars", type=int, default=4000)
    ap.add_argument("--if_max_chars", type=int, default=4000)

    ap.add_argument("--if_hard_frac", type=float, default=0.35, help="fraction of IF to be code/math-looking")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("Loading + per-class dedup…")
    fqa_all  = load_fqa(args.fqa_file,  args.fqa_max_chars)
    reas_all = load_reasoning(args.reas_file, args.reas_max_chars)
    if_all   = load_if(args.if_file,   args.if_max_chars)

    tmp = fqa_all + reas_all + if_all
    _report_counts("Loaded (after per-class dedup, before sampling)", tmp)

    print("\nMining hard IF (code/math-looking)…")
    if_hard_pool = mine_if_hard_cases(if_all, args.if_max_chars)
    print(f"  IF hard pool size: {len(if_hard_pool):,} / IF total: {len(if_all):,}")

    print("\nBalanced sampling (FQA/REAS/IF)…")
    fqa = sample_balanced(fqa_all, args.per_class, args.seed)
    reas = sample_balanced(reas_all, args.per_class, args.seed+1)

    # IF: enforce hard fraction
    target_if = args.per_class
    target_hard = int(max(0, min(1.0, args.if_hard_frac)) * target_if)

    rng.shuffle(if_hard_pool)
    if_hard = if_hard_pool[: min(len(if_hard_pool), target_hard)]

    # exclude hard cases from the easy pool using normalized-question hashes
    hard_keys = { _hash_norm(r["question"]) for r in if_hard }
    if_easy_candidates = [r for r in if_all if _hash_norm(r["question"]) not in hard_keys]

    need_easy = target_if - len(if_hard)
    rng.shuffle(if_easy_candidates)
    if_easy = if_easy_candidates[: max(0, need_easy)]

    iff = if_hard + if_easy
    rng.shuffle(iff)

    report_mid = fqa + reas + iff
    _report_counts("After balanced sampling (FQA/REAS/IF)", report_mid)

    print("\nBuilding OTHER (GSM8K + MBPP)…")
    other = build_other(args.other_total, args.seed+3)
    _report_counts("OTHER pool", other)

    print("\nMerging + global dedup + shuffle…")
    all_rows = fqa + reas + iff + other
    before = len(all_rows)
    all_rows = _dedup_keep_first(all_rows)   # dedup across classes too
    after = len(all_rows)
    if after < before:
        print(f"  Removed {before-after:,} near-duplicates across classes.")
    rng.shuffle(all_rows)
    _report_counts("Final merged set (pre-split)", all_rows)

    # write train / val
    _ensure_dir(args.out_dir)
    if 0.0 < args.val_frac < 0.5 and len(all_rows) >= 1000:
        cut = int(len(all_rows) * (1.0 - args.val_frac))
        train_rows = all_rows[:cut]
        val_rows = all_rows[cut:]
        _write_jsonl(os.path.join(args.out_dir, "router_cls_train.jsonl"), train_rows)
        _write_jsonl(os.path.join(args.out_dir, "router_cls_val.jsonl"), val_rows)
    else:
        _write_jsonl(os.path.join(args.out_dir, "router_cls_train.jsonl"), all_rows)

    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"labels": LABELS,
             "desc": {"FQA":"factual_qa", "REAS":"reasoning", "IF":"instruction_following", "OTHER":"unknown/other"}},
            f, indent=2
        )
    print("\nDone.")

if __name__ == "__main__":
    main()