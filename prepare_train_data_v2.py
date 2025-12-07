import os, json, argparse, random, re
from datasets import load_dataset

# ================== task mix config ==================
# 3-way even split across top-level task types
TASK_RATIOS = {
    "triviaqa":              1.0/3.0,
    "arc-c":                 1.0/3.0,
    "instruction_following": 1.0/3.0,
}

# Within-task weights (normalized per task)
WITHIN_TASK_WEIGHTS = {
    # Put more weight on TriviaQA vs NQ for TriviaQA eval
    "triviaqa": {"triviaqa": 0.7, "nq_open": 0.3},
    # Put more weight on ARC for ARC-C eval (CSQA/OBQA still included)
    "arc-c":    {"arc": 0.6, "csqa": 0.25, "obqa": 0.15},
    # IF eval won’t use these directly; keep clean, high-quality sets
    "instruction_following": {"tulu3": 0.8, "smoltalk": 0.2},
}

# ARC-only answer-choice shuffle augmentation (1 = off). You asked for ×5.
ARC_CHOICE_SHUFFLE_REPS = 5

# TODO: retain only 4 choices for commensenseQA, add CoT during training.
# ============================== utils ====================================
def _ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

def _write_jsonl(path, rows):
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} rows → {path}")

def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def _read_count(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def _extract_id(ex, idx, keys):
    for k in keys:
        if k in ex and ex[k] not in (None, ""):
            try:
                return str(ex[k])
            except Exception:
                pass
    return str(idx)

def _join_nonempty(parts, sep="\n\n"):
    return sep.join([p for p in parts if p and str(p).strip()])

def _mk_mcqa_question(stem, labels, texts):
    stem = str(stem).strip()
    pairs = [f"{L}) {str(T).strip()}" for L, T in zip(labels, texts)]
    return "Question: " + stem + "\n\nChoices:\n" + "\n".join(pairs)

def _normalize(d):
    s = float(sum(d.values())) or 1.0
    return {k: (v / s) for k, v in d.items()}

# ============================ TRIVIA loaders =================================
def build_triviaqa_train(out_path, subset="unfiltered.nocontext"):
    ds = load_dataset("mandarjoshi/trivia_qa", subset, split="train")
    rows = []
    for idx, ex in enumerate(ds):
        qid = _extract_id(ex, idx, ["id","qid","question_id","questionId","example_id","key"])
        q = str(ex.get("question", "")).strip()
        ans = ex.get("answer", {}) or {}
        aliases = ans.get("normalized_aliases") or []
        main_val = ans.get("normalized_value") or ans.get("value")
        golds = [str(a) for a in aliases] if aliases else ([str(main_val)] if main_val else [])
        if not q or not golds:
            continue
        rows.append({
            "id": qid,
            "task_type": "triviaqa",
            "task_name": "triviaqa",
            "question": q,
            "answers": golds,
        })
    _write_jsonl(out_path, rows)
    return out_path

def build_natural_questions_train(out_path):
    ds = load_dataset("sentence-transformers/natural-questions", split="train")
    rows = []
    for idx, ex in enumerate(ds):
        q = str(ex.get("query", "")).strip()
        a = str(ex.get("answer", "")).strip()
        if not q or not a:
            continue
        rows.append({
            "id": f"nq_{idx+1}",
            "task_type": "triviaqa",
            "task_name": "nq_open",
            "question": q,
            "answers": [a],
        })
    _write_jsonl(out_path, rows)
    return out_path

# ============================ MCQ loaders (ARC/CSQA/OBQA) ===================
def _arc_rows_from_subset(subset):
    ds = load_dataset("allenai/ai2_arc", subset, split="train")
    rows = []
    for idx, ex in enumerate(ds):
        qid = _extract_id(ex, idx, ["id","qid","question_id","questionId","example_id","key"])
        ch = ex.get("choices", {}) or {}
        labels = [str(L).strip().upper() for L in (ch.get("label", []) or [])]
        texts  = [str(T).strip()         for T in (ch.get("text",  []) or [])]
        gold   = str(ex.get("answerKey", "")).strip().upper()
        if not labels or not texts or len(labels) != len(texts) or gold not in labels:
            continue
        stem = str(ex.get("question", "")).strip()
        rows.append({
            "id": f"{subset}-{qid}",
            "task_type": "arc-c",
            "task_name": "arc",
            "question": _mk_mcqa_question(stem, labels, texts),
            "choices": {"label": labels, "text": texts},
            "answerKey": gold,  # LETTER target (your trainer uses this)
        })
    return rows

def build_arc_combined_train(out_path):
    rows = _arc_rows_from_subset("ARC-Challenge") + _arc_rows_from_subset("ARC-Easy")
    _write_jsonl(out_path, rows)
    return out_path

def build_csqa_train(out_path):
    ds = load_dataset("tau/commonsense_qa", split="train")
    rows = []
    for idx, ex in enumerate(ds):
        rid   = _extract_id(ex, idx, ["id","question_id","qid","example_id"])
        stem  = str(ex.get("question", "") or "").strip()
        ch    = ex.get("choices", {}) or {}
        labels = [str(L).strip().upper() for L in (ch.get("label", []) or [])]   # A..E
        texts  = [str(T).strip()         for T in (ch.get("text",  []) or [])]
        gold   = str(ex.get("answerKey", "") or "").strip().upper()
        if not stem or not labels or not texts or len(labels) != len(texts) or gold not in labels:
            continue
        rows.append({
            "id": f"csqa-{rid}",
            "task_type": "arc-c",
            "task_name": "csqa",
            "question": _mk_mcqa_question(stem, labels, texts),
            "choices": {"label": labels, "text": texts},
            "answerKey": gold,
            "answers": [texts[labels.index(gold)]],
        })
    _write_jsonl(out_path, rows)
    return out_path

def build_obqa_train(out_path):
    ds = load_dataset("allenai/openbookqa", "main", split="train")
    rows = []
    for idx, ex in enumerate(ds):
        rid  = _extract_id(ex, idx, ["id","question_id","qid","example_id","fact_id"])
        stem = str(ex.get("question_stem", "") or "").strip()
        ch   = ex.get("choices", {}) or {}
        labels = [str(L).strip().upper() for L in (ch.get("label", []) or [])]   # A..D
        texts  = [str(T).strip()         for T in (ch.get("text",  []) or [])]
        gold   = str(ex.get("answerKey", "") or "").strip().upper()
        if not stem or not labels or not texts or len(labels) != len(texts) or gold not in labels:
            continue
        rows.append({
            "id": f"obqa-{rid}",
            "task_type": "arc-c",
            "task_name": "obqa",
            "question": _mk_mcqa_question(stem, labels, texts),
            "choices": {"label": labels, "text": texts},
            "answerKey": gold,
            "answers": [texts[labels.index(gold)]],
        })
    _write_jsonl(out_path, rows)
    return out_path

# ============== ARC-only MCQ augmentation: shuffle answer choices ============
def augment_arc_by_choice_shuffles(in_path, out_path, reps, seed=42):
    if reps <= 1:
        return in_path
    rng = random.Random(seed)
    base = _read_jsonl(in_path)
    out  = []
    for r in base:
        out.append(r)
        ch = r.get("choices", {}) or {}
        labels = ch.get("label", []) or []
        texts  = ch.get("text",  []) or []
        gold   = (r.get("answerKey", "") or "").strip().upper()
        if not labels or not texts or len(labels) != len(texts) or gold not in labels:
            continue
        stem = r["question"].split("\n\nChoices:\n")[0].replace("Question: ", "").strip()
        gidx = labels.index(gold)
        for k in range(1, reps):
            perm = list(range(len(labels))); rng.shuffle(perm)
            new_texts = [texts[j] for j in perm]
            new_pos   = perm.index(gidx)
            new_gold  = labels[new_pos]
            out.append({
                **r,
                "id": f"{r['id']}-shuf{k}",
                "question": _mk_mcqa_question(stem, labels, new_texts),
                "answers": [new_texts[new_pos]],
                "choices": {"label": labels, "text": new_texts},
                "answerKey": new_gold,
            })
    _write_jsonl(out_path, out)
    return out_path

# ======================= IF helpers & loaders ================================
_ROLE_PREFIXES = (
    r"^\s*(User|Assistant|System)\s*:\s*",
    r"^\s*###\s*(Instruction|Input|Response)\s*:\s*",
)
_CODE_FENCE = r"^\s*```[a-zA-Z0-9]*\s*|\s*```\s*$"

def _strip_chat_markers(text):
    if not text:
        return ""
    t = text.strip()
    t = re.sub(_CODE_FENCE, "", t, flags=re.MULTILINE)
    for pat in _ROLE_PREFIXES:
        t = re.sub(pat, "", t, count=1, flags=re.IGNORECASE | re.MULTILINE).strip()
    return t

def _messages_to_if_pair(messages):
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

def build_tulu3_if_train(out_path):
    ds = load_dataset("allenai/tulu-3-sft-personas-instruction-following", split="train")
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
        rows.append({
            "id": f"tulu3-{rid}",
            "task_type": "instruction_following",
            "task_name": "tulu3",
            "question": instr,     # plain instruction only
            "answers": [target],
        })
    _write_jsonl(out_path, rows)
    return out_path

def build_smoltalk_train(out_path, subset="all"):
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
        rows.append({
            "id": f"smol-{rid}",
            "task_type": "instruction_following",
            "task_name": "smoltalk",
            "question": instr,     # plain instruction only
            "answers": [target],
        })
    _write_jsonl(out_path, rows)
    return out_path

# ======================= mixing & sampling ===================================
def _force_unique_id(row, seen, tag):
    rid = str(row.get("id", ""))
    if rid not in seen:
        seen.add(rid); return row
    base = rid or f"{tag}-ex"
    k = 1
    while True:
        alt = f"{base}#dup{k}"
        if alt not in seen:
            new_row = {**row, "id": alt}
            seen.add(alt)
            return new_row
        k += 1

def _sample_rows(rows, want, rng, tag):
    if want <= 0 or not rows:
        return []
    n = len(rows)
    if want <= n:
        picked = rng.sample(rows, want)
    else:
        picked = rows[:] + rng.choices(rows, k=want - n)  # with replacement
    seen = set()
    uniq = [_force_unique_id(r, seen, tag) for r in picked]
    return uniq

def mix_threeway(out_path, total_examples, task_ratios,
                 pools_by_task,
                 within_task_weights=None,
                 seed=42):
    rng = random.Random(seed)
    final = []
    report_lines = []

    task_ratios = _normalize(task_ratios)
    within_task_weights = within_task_weights or {}

    for task, tratio in task_ratios.items():
        want_task = int(round(total_examples * tratio))
        ds_keys = list(pools_by_task[task].keys())

        # determine within-task weights
        if task in within_task_weights:
            w = _normalize({k: v for k, v in within_task_weights[task].items() if k in ds_keys})
        else:
            sizes = {k: _read_count(pools_by_task[task][k]) for k in ds_keys}
            total_size = sum(sizes.values()) or 1
            w = {k: sizes[k] / total_size for k in ds_keys}

        added_this_task = 0
        for k in ds_keys:
            want_k = int(round(w.get(k, 0.0) * want_task))
            rows_k = _read_jsonl(pools_by_task[task][k])
            picked = _sample_rows(rows_k, want_k, rng, tag=k)
            final.extend(picked)
            added_this_task += len(picked)
            report_lines.append(f"{task:23s} | {k:14s} | avail={len(rows_k):7d} want={want_k:7d} got={len(picked):7d}")

        # rounding top-up from largest available sub-pool
        deficit = want_task - added_this_task
        if deficit > 0 and ds_keys:
            src_key = max(ds_keys, key=lambda k: _read_count(pools_by_task[task][k]))
            src_rows = _read_jsonl(pools_by_task[task][src_key])
            picked = _sample_rows(src_rows, deficit, rng, tag=src_key)
            final.extend(picked)
            report_lines.append(f"{task:23s} | topup->{src_key:8s} | deficit={deficit:4d} filled={len(picked):4d}")

    rng.shuffle(final)
    _write_jsonl(out_path, final)
    print("Mixture report:")
    for line in report_lines:
        print("  " + line)
    print(f"TOTAL mixed: {len(final):,} → {out_path}")
    return out_path

# ================================= main ======================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--total_examples", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--triviaqa_subset", default="unfiltered.nocontext")
    args = ap.parse_args()

    od = os.path.abspath(args.out_dir); _ensure_dir(od)

    p_trivia = os.path.join(od, "triviaqa_train_v2.jsonl")
    p_nq     = os.path.join(od, "nq_open_train_v2.jsonl")
    p_arc    = os.path.join(od, "arc_ec_train_v2.jsonl")
    p_csqa   = os.path.join(od, "csqa_train_v2.jsonl")
    p_obqa   = os.path.join(od, "obqa_train_v2.jsonl")
    p_tulu   = os.path.join(od, "tulu3_if_train_v2.jsonl")
    p_smol   = os.path.join(od, "smoltalk_train_v2.jsonl")

    print(f"Building TriviaQA → {p_trivia}")
    build_triviaqa_train(p_trivia, subset=args.triviaqa_subset)

    print(f"Building NQ-Open → {p_nq}")
    build_natural_questions_train(p_nq)

    print(f"Building ARC (Easy+Challenge) → {p_arc}")
    build_arc_combined_train(p_arc)

    print(f"Building CommonsenseQA → {p_csqa}")
    build_csqa_train(p_csqa)

    print(f"Building OpenBookQA → {p_obqa}")
    build_obqa_train(p_obqa)

    print(f"Building TULU3-IF → {p_tulu}")
    build_tulu3_if_train(p_tulu)

    print(f"Building SmolTalk → {p_smol}")
    build_smoltalk_train(p_smol, subset="all")

    # ---- ARC-only augmentation by shuffling answer choices ----
    if ARC_CHOICE_SHUFFLE_REPS > 1:
        p_arc_aug = os.path.join(od, f"arc_ec_train_aug{ARC_CHOICE_SHUFFLE_REPS}.jsonl")
        print(f"Augmenting ARC by choice shuffles ×{ARC_CHOICE_SHUFFLE_REPS} → {p_arc_aug}")
        p_arc = augment_arc_by_choice_shuffles(p_arc, p_arc_aug, reps=ARC_CHOICE_SHUFFLE_REPS, seed=args.seed)

    # ---- group pools by task type ----
    pools_by_task = {
        "triviaqa": {
            "triviaqa": p_trivia,
            "nq_open":  p_nq,
        },
        "arc-c": {
            "arc":   p_arc,   # already augmented
            "csqa":  p_csqa,
            "obqa":  p_obqa,
        },
        "instruction_following": {
            "tulu3":    p_tulu,
            "smoltalk": p_smol,
        },
    }

    # ---- final 3-way mixture ----
    mix_path = os.path.join(od, "sft_mixture.jsonl")
    mix_threeway(
        out_path=mix_path,
        total_examples=args.total_examples,
        task_ratios=TASK_RATIOS,
        pools_by_task=pools_by_task,
        within_task_weights=WITHIN_TASK_WEIGHTS,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
