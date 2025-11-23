import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

def build_prompt_and_target(ex):
    task = (ex.get("task_type") or "").lower()
    q = ex.get("question", "")
    if not q:
        return None

    if task == "triviaqa":
        golds = ex.get("answers") or []
        if not golds:
            return None
        target = str(golds[0]).strip()
        prompt = f"Question: {q}\nAnswer:"
        return {"prompt": prompt, "target": target}

    if task == "arc-c":
        key = str(ex.get("answerKey", "")).strip().upper()
        if key not in ("A", "B", "C", "D"):
            return None
        prompt = f"{q}\nAnswer:" # arc already formatted as "Question: {question}\n\nChoices:\n{choices}"
        target = key
        return {"prompt": prompt, "target": target}
    # Unknown task_type -> skip
    return None


def tokenize_with_prompt_mask(tokenizer, example, max_length):
    prompt = example["prompt"]
    target = example["target"]

    # append EOS so model learns to stop; if eos_token is None, skip
    eos = tokenizer.eos_token or ""
    target_ids = tokenizer(target + eos, add_special_tokens=False).input_ids

    # now tokenize prompt, truncated to leave room for target
    max_prompt_len = max(1, max_length - len(target_ids))
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_len).input_ids

    # stitch together prompt and target
    input_ids = prompt_ids + target_ids
    attn_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids # mask prompt, learn on target

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    
    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        attn_mask = attn_mask + [0] * pad_len
        labels = labels + [-100] * pad_len
    else:
        input_ids = input_ids[:max_length]
        attn_mask = attn_mask[:max_length]
        labels = labels[:max_length]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels,
    }

def load_and_prepare_dataset(train_file, tokenizer, max_length):
    print(f"\nLoading dataset from: {train_file}")
    raw = load_dataset("json", data_files={"train": train_file})["train"]
    print(f"  Raw dataset size: {len(raw):,} examples")

    # Keep only rows we can map to (prompt, target)
    def _keep_map(ex):
        pt = build_prompt_and_target(ex)
        return {"_keep": pt is not None}

    kept = raw.map(_keep_map, remove_columns=[])
    valid = raw.filter(lambda ex, idx: kept[idx]["_keep"], with_indices=True)
    num_valid = len(valid)
    num_filtered = len(raw) - num_valid
    print(f"  Valid examples: {num_valid:,} ({num_filtered:,} filtered out)")
    
    # Tokenize with masked prompts
    def _tok_map(ex):
        pt = build_prompt_and_target(ex)
        return tokenize_with_prompt_mask(tokenizer, pt, max_length)

    # drop original text cols; keep only tensors
    drop_cols = list(valid.column_names)   
    ds_train = valid.map(_tok_map, remove_columns=drop_cols, desc="Tokenizing", num_proc=1)
    print(f"  Tokenization complete: {len(ds_train):,} examples")
    return ds_train

def load_tok_and_model(model_name, fp16, bf16):
    print(f"Loading tokenizer and model from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token: {tokenizer.eos_token}")
    tokenizer.padding_side = "right"

    dtype = None
    if fp16:
        dtype = torch.float16
        print(f"  Using float16 precision")
    elif bf16:
        dtype = torch.bfloat16
        print(f"  Using bfloat16 precision")
    else:
        print(f"  Using default precision (float32)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=None
    )
    print(f"  Model loaded: {model.config.model_type} with {model.num_parameters():,} parameters")
    
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model

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
):
    set_seed(seed)
    tok, model = load_tok_and_model(model_name, fp16=fp16, bf16=bf16)
    ds_train = load_and_prepare_dataset(train_file, tok, max_length)

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
        eval_strategy="no",  # no internal holdout; use external eval.py
        save_total_limit=3,
        seed=seed,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        processing_class=tok,
        data_collator=default_data_collator,  # dynamic right padding to pad_token_id
    )

    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"Done. Model saved to: {output_dir}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, default="data/sft_data.jsonl")
    ap.add_argument("--model", type=str, default="google/gemma-3-270m")
    ap.add_argument("--output_dir", type=str, default="models/gemma270m-sft-v1")

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
    )

if __name__ == "__main__":
    main()