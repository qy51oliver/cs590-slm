import os, json, argparse, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

# -------------------- utils --------------------
def _assert_file(p, name):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Required {name} not found: {p}")

def _load_labels(train_file):
    base = os.path.dirname(os.path.abspath(train_file)) or "."
    lj = os.path.join(base, "labels.json")
    _assert_file(lj, "labels.json")
    with open(lj, "r", encoding="utf-8") as f:
        obj = json.load(f)
    labels = obj.get("labels")
    assert isinstance(labels, list) and labels == ["FQA", "REAS", "IF"], \
        f"labels.json must be exactly ['FQA','REAS','IF'], got: {labels}"
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return labels, label2id, id2label

def _assert_labels_subset(ds, allowed):
    seen = set()
    for r in ds:
        lab = (r.get("label") or "").strip()
        if lab:
            seen.add(lab)
    bad = seen - set(allowed)
    if bad:
        raise ValueError(f"Unexpected labels in data: {sorted(bad)}; allowed={allowed}")

def _print_split_stats(name, ds, labels):
    counts = {lab: 0 for lab in labels}
    for r in ds:
        lab = r.get("label")
        if lab in counts:
            counts[lab] += 1
    tot = len(ds)
    def pct(x): return 0.0 if tot == 0 else 100.0 * x / tot
    print(f"\n{name} size: {tot:,}")
    for lab in labels:
        print(f"  {lab:<4}: {counts[lab]:7,d} ({pct(counts[lab]):5.1f}%)")

# -------------------- tokenization / truncation --------------------
def _head_tail_truncate(input_ids, max_length, head_ratio=0.7):
    if len(input_ids) <= max_length:
        return input_ids
    # keep BOS if present
    head = int(max_length * head_ratio)
    tail = max_length - head
    return input_ids[:head] + input_ids[-tail:]

def _build_tokenize_fn(tokenizer, max_length, truncation_style="headtail"):
    assert truncation_style in ("head", "headtail")
    def _tok(ex):
        text = (ex.get("question") or "").strip()
        enc = tokenizer(text, add_special_tokens=True, padding=False, truncation=False)
        ids = enc["input_ids"]
        if len(ids) > max_length:
            if truncation_style == "headtail":
                ids = _head_tail_truncate(ids, max_length, head_ratio=0.7)
            else:
                ids = ids[:max_length]
        attn = [1] * len(ids)
        lab = (ex.get("label") or "").strip()
        return {"input_ids": ids, "attention_mask": attn, "label": lab}
    return _tok

def _build_label_map_fn(label2id):
    def _lab(ex):
        lab = (ex.get("label") or "").strip()
        ex["labels"] = label2id[lab]  # strict: KeyError if unseen
        ex.pop("label", None)
        return ex
    return _lab

# -------------------- metrics --------------------
def _compute_metrics_builder(id2label):
    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = float((preds == labels).mean())
        per_class = {}
        for i, lab in id2label.items():
            mask = (labels == i)
            denom = int(mask.sum())
            per_class[f"acc_{lab}"] = 0.0 if denom == 0 else float(((preds == i) & mask).sum()) / denom
        out = {"accuracy": acc}
        out.update(per_class)
        return out
    return _compute

# -------------------- main train --------------------
def train(
    train_file,
    val_file,
    model_name,
    output_dir,
    max_length=1024,                     
    truncation_style="headtail",          
    per_device_train_batch_size=8,      
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,      
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    seed=42,
    fp16=False,
    bf16=True,
):
    _assert_file(train_file, "train_file")
    _assert_file(val_file, "val_file")

    print("Loading label spec…")
    labels, label2id, id2label = _load_labels(train_file)
    print(f"Labels: {labels}")
    print(f"label2id: {label2id}")

    print("\nLoading splits…")
    ds = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    dtrain_raw = ds["train"]
    dval_raw = ds["validation"]

    for col in ("id", "question", "label"):
        assert col in dtrain_raw.column_names, f"train split missing column: {col}"
        assert col in dval_raw.column_names, f"validation split missing column: {col}"

    _assert_labels_subset(dtrain_raw, labels)
    _assert_labels_subset(dval_raw, labels)
    _print_split_stats("Train", dtrain_raw, labels)
    _print_split_stats("Validation", dval_raw, labels)

    print("\nLoading tokenizer/model…")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print(f"  Set pad_token -> eos_token: {tok.eos_token}")
    tok.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        trust_remote_code=True,
    )
    model.config.label2id = {k: int(v) for k, v in label2id.items()}
    model.config.id2label = {int(k): v for k, v in id2label.items()}
    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id

    print("\nTokenizing…")
    tok_fn = _build_tokenize_fn(tok, max_length, truncation_style=truncation_style)
    lab_fn = _build_label_map_fn(label2id)

    dtrain_tok = dtrain_raw.map(tok_fn, remove_columns=[])
    dval_tok   = dval_raw.map(tok_fn,   remove_columns=[])

    dtrain = dtrain_tok.map(lab_fn, remove_columns=[c for c in dtrain_tok.column_names if c not in ("input_ids","attention_mask","labels")])
    dval   = dval_tok.map(lab_fn,   remove_columns=[c for c in dval_tok.column_names   if c not in ("input_ids","attention_mask","labels")])

    print(f"Tokenized sizes → train: {len(dtrain):,}, val: {len(dval):,}")

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=seed,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=False,
        dataloader_num_workers=2,
        optim="adamw_torch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dtrain,
        eval_dataset=dval,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8),
        compute_metrics=_compute_metrics_builder(id2label),
    )

    print("\nTraining…")
    trainer.train()
    print("\nSaving best to root…")
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"Done. Best checkpoint merged to: {output_dir}")

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train a strict 3-way router classifier (FQA/REAS/IF) with no fallbacks.")
    # Deterministic defaults for your repo layout
    ap.add_argument("--train_file", default="data/router_cls/router_cls_train.jsonl", type=str)
    ap.add_argument("--val_file",   default="data/router_cls/router_cls_val.jsonl", type=str)
    ap.add_argument("--model",      default="google/gemma-3-270m", type=str)
    ap.add_argument("--output_dir", default="models/router-cls-gemma270m", type=str)

    # Longer default length; head+tail keeps both task cues and options
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--truncation_style", type=str, default="headtail", choices=["head","headtail"])

    # Safe-by-default batch for 1024 ctx; tune if you have headroom
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--per_device_eval_batch_size",  type=int, default=64)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)

    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true", default=True)
    return ap.parse_args()

def main():
    args = parse_args()
    train(
        train_file=args.train_file,
        val_file=args.val_file,
        model_name=args.model,
        output_dir=args.output_dir,
        max_length=args.max_length,
        truncation_style=args.truncation_style,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
    )

if __name__ == "__main__":
    main()