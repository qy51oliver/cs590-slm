import os, argparse, re, torch
from typing import Any, Dict, List
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
def _apply_chat_template(tokenizer, messages_or_str, use_chat_template: bool) -> str:
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

def build_prompt_and_target(ex: Dict[str, Any], tokenizer, use_chat_template: bool):
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

    # instruction_following  (pipeline feeds plain instruction through chat template)
    if task in ("instruction_following", "ifeval"):
        answers = ex.get("answers") or []
        if not answers:
            return None
        target = str(answers[0]).strip()
        prompt = _apply_chat_template(tokenizer, q, use_chat_template)
        return {"prompt": prompt, "target": target}

    return None

# ---------------- tokenization (mask prompt, learn target) ----------------
def tokenize_with_prompt_mask(tokenizer, example: Dict[str, str], max_length: int):
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
def load_and_prepare_dataset(train_file: str, tokenizer, max_length: int, use_chat_template: bool):
    print(f"\nLoading dataset from: {train_file}")
    raw = load_dataset("json", data_files={"train": train_file})["train"]
    print(f"  Raw dataset size: {len(raw):,}")

    def _keep_map(ex):
        pt = build_prompt_and_target(ex, tokenizer, use_chat_template)
        return {"_keep": pt is not None}

    kept = raw.map(_keep_map, remove_columns=[])
    valid = raw.filter(lambda ex, idx: kept[idx]["_keep"], with_indices=True)
    print(f"  Valid examples after task-specific filtering: {len(valid):,} (dropped {len(raw)-len(valid):,})")

    def _tok_map(ex):
        pt = build_prompt_and_target(ex, tokenizer, use_chat_template)
        return tokenize_with_prompt_mask(tokenizer, pt, max_length)

    drop_cols = list(valid.column_names)
    ds_train = valid.map(_tok_map, remove_columns=drop_cols, desc="Tokenizing", num_proc=1)
    # after building ds_train, add:
    too_long = sum(len(ex["labels"]) == max_length for ex in ds_train)
    print(f"  Truncation-at-max_length examples: {too_long:,} (consider raising --max_length)")
    return ds_train

def load_tok_and_model(model_name: str):
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
    train_file: str,
    model_name: str,
    output_dir: str,
    max_length: int = 512,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    num_train_epochs: int = 2,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    seed: int = 42,
    fp16: bool = True,
    bf16: bool = False,
    use_chat_template: bool = True,   # <- default ON to mirror pipeline
):
    set_seed(seed)
    tok, model = load_tok_and_model(model_name)
    ds_train = load_and_prepare_dataset(train_file, tok, max_length, use_chat_template)

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
        save_strategy="epoch",
        eval_strategy="no",
        save_total_limit=2,
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
        processing_class=tok,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"Done. Model saved to: {output_dir}")

# ---------------- cli ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--model", type=str, default="google/gemma-3-270m")
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no_chat_template", action="store_true",
                    help="Disable applying tokenizer.chat_template during training.")
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
    )

if __name__ == "__main__":
    main()
