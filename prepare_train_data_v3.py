import os, json, math, argparse, random, re
from datasets import load_dataset

# ------------------------- utils -------------------------
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _write_jsonl(path, rows):
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} rows → {path}")

def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def _extract_id(ex, idx, candidates):
    """Return first available id-like field as string, else the index."""
    for k in candidates:
        if k in ex and ex[k] not in (None, ""):
            try:
                return str(ex[k])
            except Exception:
                pass
    return str(idx)

def _merge_and_write(paths, out_path, shuffle=True, seed=42):
    rng = random.Random(seed)
    merged = []
    for p in paths:
        merged.extend(_read_jsonl(p))
    if shuffle:
        rng.shuffle(merged)
    _write_jsonl(out_path, merged)
    return out_path

def _mk_mcqa_question(stem, labels, texts):
    stem = (stem or "").strip()
    pairs = [f"{L}) {str(T).strip()}" for L, T in zip(labels, texts)]
    return "Question: " + stem + "\n\nChoices:\n" + "\n".join(pairs)

def _csqa_to_four(labels, texts, gold, rng):
    # keep gold + 3 random distractors; reletter to A–D
    assert gold in labels
    gold_idx = labels.index(gold)
    idxs = list(range(len(labels)))
    idxs.remove(gold_idx)
    rng.shuffle(idxs)
    keep = [gold_idx] + idxs[:3]
    keep.sort()
    new_texts = [texts[i] for i in keep]
    new_labels = ["A","B","C","D"]
    # gold moves to new_labels[new_pos]
    new_pos = keep.index(gold_idx)
    new_gold = new_labels[new_pos]
    return new_labels, new_texts, new_gold

# -------- IF helpers (strip any chat-y markers; keep plain instr/resp) ------
_ROLE_PREFIXES = (
    r"^\s*(User|Assistant|System)\s*:\s*",
    r"^\s*###\s*(Instruction|Input|Response)\s*:\s*",
)
_CODE_FENCE = r"^\s*```[a-zA-Z0-9]*\s*|\s*```\s*$"

def _strip_chat_markers(text):
    if not text:
        return ""
    t = text
    t = re.sub(_CODE_FENCE, "", t, flags=re.MULTILINE)
    for pat in _ROLE_PREFIXES:
        t = re.sub(pat, "", t, count=1, flags=re.IGNORECASE | re.MULTILINE).strip()
    return t.strip()

def _join_nonempty(parts, sep="\n\n"):
    return sep.join([p for p in parts if p and str(p).strip()])

def _messages_to_if_pair(messages):
    """Flatten chat messages into (instruction_text, target_text)."""
    instr_parts, target = [], None
    for m in (messages or []):
        role = (m.get("role") or "").strip().lower()
        content = _strip_chat_markers(str(m.get("content", "") or ""))
        if not content:
            continue
        if role == "assistant":
            if target is None:
                target = content
        else:
            instr_parts.append(content)
    return _join_nonempty(instr_parts), (target or "")

CONSTRAINT_REGEX = re.compile(
    r"\b("
    r"exactly|at\s+least|no\s+more\s+than|between|at\s+most|"
    r"include|must\s+include|exclude|forbid|avoid|"
    r"start\s+with|end\s+with|first\s+word|last\s+word|"
    r"json|yaml|xml|markdown\s+table|bullets?|sections?|"
    r"paragraphs?|sentences?|words?|characters?|"
    r"uppercase|lowercase|title\s*case|capitalization|"
    r"roman\s+numerals|alphabetical\s+order|numbered\s+list"
    r")\b",
    re.IGNORECASE
)

def _looks_constraint_like(text):
    return bool(CONSTRAINT_REGEX.search(text or ""))

def _make_words(n):
    base = "lorem ipsum dolor sit amet consectetur adipiscing elit"
    words = (base.split() * ((n // 7) + 2))[:n]
    return " ".join(words)

# ---------- dataset builders ----------

# =============== TRIVIA ==================
def download_triviaqa_train(out_path, subset="unfiltered.nocontext"):
    ds = load_dataset("mandarjoshi/trivia_qa", subset, split="train")
    rows = []
    for idx, ex in enumerate(ds):
        qid = _extract_id(ex, idx, candidates=["id","qid","question_id","questionId","example_id","key"])
        q = ex.get("question", "")
        ans = ex.get("answer", {}) or {}
        aliases = ans.get("normalized_aliases") or []
        main_val = ans.get("normalized_value") or ans.get("value")
        golds = [str(a) for a in aliases] if aliases else ([str(main_val)] if main_val else [])
        rows.append({
            "id": qid,
            "task_type": "factual_qa",
            "question": str(q),
            "answers": golds,   # list; your trainer will pick the first
        })
    _write_jsonl(out_path, rows)
    return out_path

# ============================ MCQ (ARC/CSQA/OBQA) ===========================
def arc_rows_from_subset(subset):
    ds = load_dataset("allenai/ai2_arc", subset, split="train")
    rows = []
    for idx, ex in enumerate(ds):
        qid = _extract_id(ex, idx, ["id","qid","question_id","questionId","example_id","key"])
        ch = ex.get("choices", {}) or {}
        labels = list(ch.get("label", []) or [])
        texts  = list(ch.get("text",  []) or [])
        gold   = str(ex.get("answerKey", "")).strip().upper()
        if not labels or not texts or gold not in labels:
            continue
        question = _mk_mcqa_question(str(ex.get("question", "")), labels, texts)
        rows.append({
            "id": qid,                        # keep ARC ids as-is
            "task_type": "reasoning",
            "task_name": "arc",
            "question": question,             # plain stem+choices
            "answerKey": gold,                # letter target
            "choices": {"label": labels, "text": texts},
        })
    return rows

def download_arc_both_train(out_path):
    rows = arc_rows_from_subset("ARC-Challenge") + arc_rows_from_subset("ARC-Easy")
    _write_jsonl(out_path, rows)
    return out_path

def download_arc_c_train(out_path):
    rows = arc_rows_from_subset("ARC-Challenge")
    _write_jsonl(out_path, rows)
    return out_path

def download_arc_e_train(out_path):
    rows = arc_rows_from_subset("ARC-Easy")
    _write_jsonl(out_path, rows)
    return out_path


def build_csqa_train(out_path, seed=42):
    ds = load_dataset("tau/commonsense_qa", split="train")
    rng = random.Random(seed)
    rows = []
    for idx, ex in enumerate(ds):
        rid = _extract_id(ex, idx, ["id","question_id","qid","example_id"])
        stem = str(ex.get("question", "") or "")
        ch   = ex.get("choices", {}) or {}
        labels = [str(L).strip().upper() for L in (ch.get("label", []) or [])]
        texts  = [str(T).strip()         for T in (ch.get("text",  []) or [])]
        gold   = str(ex.get("answerKey", "") or "").strip().upper()
        if not stem or not labels or not texts or gold not in labels:
            continue

        # --- normalize to 4 options ---
        if len(labels) == 5:
            labels, texts, gold = _csqa_to_four(labels, texts, gold, rng)
        elif len(labels) != 4:
            continue  # skip odd cases

        rows.append({
            "id": f"csqa-{rid}",
            "task_type": "reasoning",
            "task_name": "commonsense_qa",
            "question": _mk_mcqa_question(stem, labels, texts),
            "answerKey": gold,
            "choices": {"label": labels, "text": texts},
            "answers": [texts[["A","B","C","D"].index(gold)]],
        })
    _write_jsonl(out_path, rows); return out_path

def build_obqa_train(out_path):
    ds = load_dataset("allenai/openbookqa", "main", split="train")
    rows = []
    for idx, ex in enumerate(ds):
        rid   = _extract_id(ex, idx, ["id","question_id","qid","example_id","fact_id"])
        stem  = str(ex.get("question_stem", "") or "")
        ch    = ex.get("choices", {}) or {}
        labels = [str(L).strip().upper() for L in (ch.get("label", []) or [])]   # A..D
        texts  = [str(T).strip()         for T in (ch.get("text",  []) or [])]
        gold   = str(ex.get("answerKey", "") or "").strip().upper()
        if not stem or not labels or not texts or gold not in labels:
            continue
        rows.append({
            "id": f"obqa-{rid}",
            "task_type": "reasoning",
            "task_name": "obqa",
            "question": _mk_mcqa_question(stem, labels, texts),
            "answerKey": gold,
            "choices": {"label": labels, "text": texts},
            "answers": [texts[labels.index(gold)]],
        })
    _write_jsonl(out_path, rows)
    return out_path

# ---------------- ARC oversampling ---------------- #
def oversample_arc_rows_with_shuffles(rows, shuffle_reps, seed):
    """Label-preserving choice-order shuffles to increase ARC data (4-choice items only)."""
    if shuffle_reps <= 1:
        return rows
    rng = random.Random(seed)
    out = []
    for r in rows:
        out.append(r)
        ch = r.get("choices", {}) or {}
        labels = ch.get("label", [])
        texts  = ch.get("text", [])
        gold   = (r.get("answerKey", "") or "").strip()
        if len(labels) != 4 or len(texts) != 4 or gold not in labels:
            continue
        stem = r["question"].split("\n\nChoices:\n")[0].replace("Question: ", "").strip()
        gold_idx = labels.index(gold)
        for k in range(1, shuffle_reps):
            perm = list(range(4)); rng.shuffle(perm)
            new_texts = [texts[j] for j in perm]
            new_pos   = perm.index(gold_idx)
            new_gold  = labels[new_pos]
            question_fmt = _mk_mcqa_question(stem, labels, new_texts)
            out.append({
                "id": f"{r['id']}-shuf{k}",
                "task_type": r.get("task_type", "reasoning"),
                "task_name": r.get("task_name", "arc"),
                "question": question_fmt,
                "answerKey": new_gold,
                "choices": {"label": labels, "text": new_texts},
                "answers": [new_texts[new_pos]],
            })
    return out

def oversample_arc_file(in_path, out_path, shuffle_reps, seed=42):
    rows = _read_jsonl(in_path)
    rows_aug = oversample_arc_rows_with_shuffles(rows, shuffle_reps, seed)
    _write_jsonl(out_path, rows_aug)
    return out_path

# ======================= IF (TULU-3 IF / SmolTalk) ==========================
def build_tulu3_if_train(out_path, *, max_examples=None, filter_constraints=False, seed=42):
    ds = load_dataset("allenai/tulu-3-sft-personas-instruction-following", split="train")
    rng = random.Random(seed)
    rows = []
    for idx, ex in enumerate(ds):
        rid = str(ex.get("id") or idx)
        messages = ex.get("messages", []) or []
        prompt_fallback = _strip_chat_markers(str(ex.get("prompt", "") or ""))
        instr, target = _messages_to_if_pair(messages)
        if not target:
            continue
        if not instr:
            instr = prompt_fallback
        if filter_constraints and not _looks_constraint_like(instr):
            continue
        rows.append({
            "id": f"tulu3-{rid}",
            "task_type": "instruction_following",
            "task_name": "tulu3",
            "question": instr,
            "answers": [target],
        })
        if max_examples and len(rows) >= max_examples:
            break
    _write_jsonl(out_path, rows); return out_path

def build_smoltalk_train(out_path, subset="all", *, max_examples=None, filter_constraints=False, seed=42):
    ds = load_dataset("HuggingFaceTB/smoltalk", subset, split="train")
    rows = []
    for idx, ex in enumerate(ds):
        rid = str(ex.get("id") or ex.get("example_id") or idx)
        messages = ex.get("messages")
        if messages:
            instr, target = _messages_to_if_pair(messages)
        else:
            prompt = _strip_chat_markers(str(ex.get("prompt", "") or ex.get("instruction", "") or ""))
            target = _strip_chat_markers(str(ex.get("response", "") or ex.get("output", "") or ""))
            instr  = prompt
        if not target:
            continue
        if filter_constraints and not _looks_constraint_like(instr):
            continue
        rows.append({
            "id": f"smol-{rid}",
            "task_type": "instruction_following",
            "task_name": "smoltalk",
            "question": instr,
            "answers": [target],
        })
        if max_examples and len(rows) >= max_examples:
            break
    _write_jsonl(out_path, rows); return out_path

def build_synthetic_ifeval_like(out_path, n=20000, seed=42):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        kind = rng.choice(["eq_words", "min_words", "bullets", "json", "start_with", "include_forbid"])
        if kind == "eq_words":
            k = rng.choice([50, 100, 150, 200])
            instr = f"Write a paragraph of **exactly {k} words** about the benefits of daily walks."
            target = _make_words(k)
        elif kind == "min_words":
            k = rng.choice([120, 180, 240])
            instr = f"Write an essay of **at least {k} words** about time management."
            target = _make_words(k + 10)
        elif kind == "bullets":
            b = rng.choice([5, 7, 9])
            instr = f"List **{b} bullet points** with tips for healthy sleep. Use '-' bullets."
            target = "\n".join(["- " + _make_words(8) for _ in range(b)])
        elif kind == "json":
            instr = "Return **valid JSON** with keys: title (string), steps (array of 3 strings). No extra text."
            target = json.dumps({"title": "Simple Pasta", "steps": ["boil water", "cook pasta", "drain and serve"]})
        elif kind == "start_with":
            w = rng.choice(["Therefore,", "Hence,", "Conclusively,"])
            instr = f"Write a short paragraph about study habits that **starts with '{w}'**."
            target = f"{w} " + _make_words(60)
        else:  # include_forbid
            inc = rng.choice(["focus", "sleep"])
            forb = rng.choice(["coffee", "sugar"])
            instr = f"Write two paragraphs about productivity that **include '{inc}'** and **do not use '{forb}'**."
            para = _make_words(70)
            # make sure to include/inc and avoid/forb:
            para = f"{para} {inc} {inc}"
            target = para + "\n\n" + _make_words(70)
        rows.append({
            "id": f"synth-{i+1}",
            "task_type": "instruction_following",
            "task_name": "ifeval_synth",
            "question": instr,
            "answers": [target],
        })
    _write_jsonl(out_path, rows); return out_path

# ---------------- main ---------------- #
def main():
    ap = argparse.ArgumentParser(description="Prepare separate per-task training datasets (model-agnostic).")
    ap.add_argument("--out-dir", default="data/v3", help="Output directory")
    ap.add_argument("--triviaqa-subset", default="unfiltered.nocontext", help="HF subset for TriviaQA")
    ap.add_argument("--arc-mode", choices=["combined","separate"], default="combined",
                    help="Write ARC Easy+Challenge combined file (default) or separate files.")
    ap.add_argument("--arc-shuffle-reps", type=int, default=5, help="ARC oversampling (1=off).")
    ap.add_argument("--if-filter-constraints", action="store_true")
    ap.add_argument("--if-max-tulu", type=int, default=30000)
    ap.add_argument("--if-max-smol", type=int, default=20000)
    ap.add_argument("--if-synth", type=int, default=0)
    
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(args.out_dir)

    # ---- Trivia head ----
    triviaqa_path = os.path.join(out_dir, "v3_triviaqa_train.jsonl")
    print(f"[Trivia] Downloading TriviaQA (train) → {triviaqa_path}")
    download_triviaqa_train(triviaqa_path, subset=args.triviaqa_subset)

    # ---- ARC head ----
    arc_path      = os.path.join(out_dir, "v3_arc_e_c_train.jsonl")
    arc_aug_path  = os.path.join(out_dir, "v3_arc_e_c_aug_train.jsonl")
    arc_c_path    = os.path.join(out_dir, "v3_arc_c_train.jsonl")
    arc_e_path    = os.path.join(out_dir, "v3_arc_e_train.jsonl")
    csqa_path     = os.path.join(out_dir, "v3_csqa_train.jsonl")
    obqa_path     = os.path.join(out_dir, "v3_obqa_train.jsonl")

    print(f"[ARC] Downloading ARC-Easy+Challenge (train) → {arc_path}")
    download_arc_both_train(arc_path)

    if args.arc_mode == "separate":
        print(f"[ARC] Also writing separate subsets…")
        download_arc_c_train(arc_c_path)
        download_arc_e_train(arc_e_path)

    if args.arc_shuffle_reps > 1:
        print(f"[ARC] Oversampling ARC (choice shuffle x{args.arc_shuffle_reps}) → {arc_aug_path}")
        oversample_arc_file(arc_path, arc_aug_path, args.arc_shuffle_reps)

    print(f"[ARC] Downloading CommonsenseQA → {csqa_path}")
    build_csqa_train(csqa_path)

    print(f"[ARC] Downloading OpenBookQA → {obqa_path}")
    build_obqa_train(obqa_path)

    arc_to_merge = arc_aug_path if args.arc_shuffle_reps > 1 else arc_path
    arc_head = os.path.join(out_dir, "v3_reasoning_train.jsonl")
    print(f"[ARC] Merging ARC(+aug?) + CSQA + OBQA → {arc_head}")
    _merge_and_write([arc_to_merge, csqa_path, obqa_path], arc_head, shuffle=True, seed=42)

    # ---- IF head ----
    tulu_path = os.path.join(out_dir, "v3_tulu3_if_train.jsonl")
    smol_path = os.path.join(out_dir, "v3_smoltalk_train.jsonl")

    print(f"[IF] Downloading TULU-3 IF → {tulu_path}")
    build_tulu3_if_train(
        tulu_path,
        max_examples=args.if_max_tulu,
        filter_constraints=args.if_filter_constraints,
    )

    print(f"[IF] Downloading SmolTalk → {smol_path}")
    build_smoltalk_train(
        smol_path,
        subset="all",
        max_examples=args.if_max_smol,
        filter_constraints=args.if_filter_constraints,
    )

    if_head = os.path.join(out_dir, "v3_instruction_following_train.jsonl")
    synth_path = None
    if args.if_synth > 0:
        synth_path = os.path.join(out_dir, "v3_ifeval_synth_train.jsonl")
        print(f"[IF] Building synthetic IF-Eval-like → {synth_path} ({args.if_synth} examples)")
        build_synthetic_ifeval_like(synth_path, n=args.if_synth)

    merge_inputs = [tulu_path, smol_path] + ([synth_path] if synth_path else [])
    print(f"[IF] Merging → {if_head}")
    _merge_and_write(merge_inputs, if_head, shuffle=True, seed=42)
        

    # ---- Trivia head “as-is” (single source) ----
    trivia_head = os.path.join(out_dir, "v3_factual_qa_train.jsonl")
    print(f"[Trivia] Writing Trivia head (copy of TriviaQA train) → {trivia_head}")
    _merge_and_write([triviaqa_path], trivia_head, shuffle=False)

    # ---- counts ----
    def _count(p): return len(_read_jsonl(p))
    print("\n=== Final per-task counts ===")
    print(f"Trivia head: {_count(trivia_head):,}  ({trivia_head})")
    print(f"ARC head:    {_count(arc_head):,}     ({arc_head})")
    print(f"IF head:     {_count(if_head):,}      ({if_head})")
    if args.arc_mode == "separate":
        print(f"ARC-C only:  {_count(arc_c_path):,}  ({arc_c_path})")
        print(f"ARC-E only:  {_count(arc_e_path):,}  ({arc_e_path})")
    if args.arc_shuffle_reps > 1:
        print(f"ARC (aug):   {_count(arc_aug_path):,}  ({arc_aug_path})")

    print("Done.")

if __name__ == "__main__":
    main()
