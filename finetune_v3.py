import os, argparse, re, torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# ---------------- prompt builders  ----------------
def _apply_chat_template(tokenizer, messages_or_str, use_chat_template):
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        msgs = messages_or_str if isinstance(messages_or_str, list) else [{"role":"user","content":str(messages_or_str)}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if isinstance(messages_or_str, list):
        lines = []
        for m in messages_or_str:
            role = m.get("role","user").lower()
            prefix = "User" if role=="user" else ("Assistant" if role=="assistant" else role.capitalize())
            lines.append(f"{prefix}: {m.get('content','')}")
        return "\n".join(lines)
    return str(messages_or_str)

def build_prompt_and_target(ex, tokenizer, use_chat_template):
    task = (ex.get("task_type") or "").lower()
    q = ex.get("question", "") or ex.get("query", "")
    if not q:
        return None

    # factual_qa 
    if task in ("factual_qa", "triviaqa"):
        golds = ex.get("answers") or []
        if not golds:
            return None
        target = str(golds[0]).strip()
        user_msg = f"Answer the question concisely.\nQuestion: {q}\nAnswer:"
        prompt = _apply_chat_template(tokenizer, [{"role":"user","content":user_msg}], use_chat_template)
        return {"prompt": prompt, "target": target}

    # reasoning
    if task in ("reasoning", "arc-c"):
        key = (ex.get("answerKey") or "").strip().upper()
        instr = (
            "You will be given a multiple-choice question with options A–D.\n"
            "Please reason and choose the correct answer **strictly as a single capital letter** (A, B, C, D). "
            "for the following question:\n"
            f"{q}\n\nAnswer:"
        )
        prompt = _apply_chat_template(tokenizer, [{"role":"user","content":instr}], use_chat_template)
        target = key
        return {"prompt": prompt, "target": target}

    # instruction_following
    if task in ("instruction_following", "ifeval"):
        answers = ex.get("answers") or []
        if not answers:
            return None
        target = str(answers[0]).strip()
        prompt = _apply_chat_template(tokenizer, q, use_chat_template)
        return {"prompt": prompt, "target": target}

    return None

# ---------------- tokenization (mask prompt, learn target) ----------------
def tokenize_with_prompt_mask(tokenizer, example, max_length):
    prompt = example["prompt"]
    target = example["target"]

    eos = tokenizer.eos_token or ""
    target_ids = tokenizer(target + eos, add_special_tokens=False).input_ids

    max_prompt_len = max(1, max_length - len(target_ids))
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_len).input_ids

    input_ids = prompt_ids + target_ids
    attn_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids += [pad_id] * pad_len
        attn_mask += [0] * pad_len
        labels += [-100] * pad_len
    else:
        input_ids = input_ids[:max_length]
        attn_mask = attn_mask[:max_length]
        labels = labels[:max_length]

    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

# ---------------- data + model loaders ----------------
def load_and_prepare_datasets(
    train_file,
    tokenizer,
    max_length,
    use_chat_template,
    val_fraction,
    seed,
):
    print(f"\nLoading dataset from: {train_file}")
    raw = load_dataset("json", data_files={"train": train_file})["train"]
    print(f"  Raw dataset size: {len(raw):,}")

    # Keep only rows we can map to (prompt, target)
    def _keep_map(ex):
        pt = build_prompt_and_target(ex, tokenizer, use_chat_template)
        return {"_keep": pt is not None}

    kept = raw.map(_keep_map, remove_columns=[])
    valid = raw.filter(lambda ex, idx: kept[idx]["_keep"], with_indices=True)
    print(f"  Valid examples after task-specific filtering: {len(valid):,} (dropped {len(raw)-len(valid):,})")

    # Split BEFORE tokenization to avoid any leakage/bias
    val_fraction = max(0.0, min(0.5, float(val_fraction)))
    if val_fraction > 0.0 and len(valid) >= 10:
        split = valid.train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
        dtrain, deval = split["train"], split["test"]
    else:
        dtrain, deval = valid, None

    print(f"  Train split: {len(dtrain):,}")
    if deval is not None:
        print(f"  Val split:   {len(deval):,}")
    else:
        print("  Val split:   (disabled)")

    def _tok_map(ex):
        pt = build_prompt_and_target(ex, tokenizer, use_chat_template)
        return tokenize_with_prompt_mask(tokenizer, pt, max_length)

    drop_cols_tr = list(dtrain.column_names)
    ds_train = dtrain.map(_tok_map, remove_columns=drop_cols_tr, desc="Tokenizing train", num_proc=1)

    ds_eval = None
    if deval is not None:
        drop_cols_ev = list(deval.column_names)
        ds_eval = deval.map(_tok_map, remove_columns=drop_cols_ev, desc="Tokenizing eval", num_proc=1)

    return ds_train, ds_eval

def load_tok_and_model(model_name):
    print(f"Loading tokenizer and model from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token: {tokenizer.eos_token}")
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # off for training
    return tokenizer, model

# ---------------- train ----------------
def train(
    train_file,
    model_name,
    output_dir,
    max_length = 512,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 4,
    num_train_epochs = 3,
    learning_rate = 2e-4,
    weight_decay = 0.01,
    warmup_ratio = 0.03,
    lr_scheduler_type = "cosine",
    seed = 42,
    fp16 = False,
    bf16 = True,
    use_chat_template = True,
    val_fraction = 0.10,
    eval_strategy = "epoch",  
    save_strategy = "epoch",  
):
    set_seed(seed)
    tok, model = load_tok_and_model(model_name)
    ds_train, ds_eval = load_and_prepare_datasets(
        train_file, tok, max_length, use_chat_template, val_fraction, seed
    )

    os.makedirs(output_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=50,
        save_strategy=save_strategy,
        eval_strategy=eval_strategy if ds_eval is not None else "no",
        save_total_limit=3,
        load_best_model_at_end=(ds_eval is not None),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        optim="adamw_torch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        processing_class=tok,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"Done. Best (by eval_loss) saved to: {output_dir}")

# ---------------- cli ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--model", type=str, default="google/gemma-3-270m")
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no_chat_template", action="store_true",
                    help="Disable applying tokenizer.chat_template during training.")

    ap.add_argument("--val_fraction", type=float, default=0.10,
                    help="Fraction (0..0.5) of data reserved for validation.")
    ap.add_argument("--eval_strategy", type=str, default="epoch",
                    help="How often to run eval/saves when val data is present.")
    return ap.parse_args()

def main():
    args = parse_args()
    train(
        train_file=args.train_file,
        model_name=args.model,
        output_dir=args.output_dir,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        use_chat_template=not args.no_chat_template,
        val_fraction=args.val_fraction,
        eval_strategy=args.eval_strategy,
    )

if __name__ == "__main__":
    main()
