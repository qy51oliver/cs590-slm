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

# ============== IT-specific prompt builders ===================

def _resolve_it_prompt_text(row, tokenizer):
    """
    Prefer chat 'messages' with tokenizer.apply_chat_template(add_generation_prompt=True).
    Fallbacks:
      - 'question' field (already like 'User: ...\\nAssistant:')
      - constructed from 'instruction' / 'context'
    """
    msgs = row.get("messages")
    if msgs and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    q = row.get("question")
    if isinstance(q, str) and q.strip():
        # Ensure it ends with the Assistant cue
        if not (q.endswith("Assistant:") or q.endswith("Assistant:\n")):
            q = q.rstrip() + "\nAssistant:"
        return q

    instr = (row.get("instruction") or "").strip()
    ctx = (row.get("context") or "").strip()
    if instr:
        if ctx:
            return f"### Instruction:\n{instr}\n\n### Input:\n{ctx}\n\n### Response:\n"
        return f"### Instruction:\n{instr}\n\n### Response:\n"

    # Last fallback: prompt field if present (string)
    p = row.get("prompt")
    if isinstance(p, str) and p.strip():
        if not (p.endswith("Assistant:") or p.endswith("Assistant:\n")):
            p = p.rstrip() + "\nAssistant:"
        return p

    return None


def _build_it_prompt_and_target(row, tokenizer):
    """Return {'prompt': str, 'target': str} or None to skip."""
    answers = row.get("answers") or []
    target = (answers[0] if answers else "").strip()
    if not target:
        return None  # skip if no answer

    prompt = _resolve_it_prompt_text(row, tokenizer)
    if not prompt or not prompt.strip():
        return None

    return {"prompt": prompt, "target": target}


def _tokenize_it_pair(tokenizer, example, max_length):
    """
    Keep full target; tail-truncate prompt so (prompt + target + EOS) <= max_length.
    Mask prompt tokens (-100) for completion-only loss.
    """
    prompt = example["prompt"]
    target = example["target"]

    eos_token = tokenizer.eos_token or ""
    # Tokenize target first to reserve space
    target_ids = tokenizer(target + eos_token, add_special_tokens=False).input_ids

    # Now tokenize prompt but tail-truncate only the prompt
    max_prompt_len = max(1, max_length - len(target_ids))
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]  # keep the tail so the 'Assistant:' cue remains

    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids  # completion-only loss

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids += [pad_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len
    else:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# ========================= Model & Dataset ================================

def load_tok_and_model(model_name):
    print(f"Loading tokenizer and model from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token: {tokenizer.eos_token}")
    tokenizer.padding_side = "right"  # CHANGED: right padding for training

    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"  Model loaded: {model.config.model_type} with {model.num_parameters():,} parameters")

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False  # keep off during training
    return tokenizer, model


def load_and_prepare_it_dataset(train_file, tokenizer, max_length, split_name="train"):
    print(f"\nLoading IT dataset [{split_name}] from: {train_file}")
    raw = load_dataset("json", data_files={split_name: train_file})[split_name]
    print(f"  Raw dataset size: {len(raw):,} examples")

    def _keep_map(ex):
        pt = _build_it_prompt_and_target(ex, tokenizer)
        return {"_keep": pt is not None}

    kept = raw.map(_keep_map, remove_columns=[])
    valid = raw.filter(lambda ex, idx: kept[idx]["_keep"], with_indices=True)
    num_valid = len(valid)
    num_filtered = len(raw) - num_valid
    print(f"  Valid examples: {num_valid:,} ({num_filtered:,} filtered out)")

    # Tokenize with masked prompts 
    def _tok_map(ex):
        pt = _build_it_prompt_and_target(ex, tokenizer)
        return _tokenize_it_pair(tokenizer, pt, max_length)

    drop_cols = list(valid.column_names)
    ds = valid.map(_tok_map, remove_columns=drop_cols, desc=f"Tokenizing ({split_name})", num_proc=1)
    print(f"  Tokenization complete: {len(ds):,} examples")
    return ds

# ============================= Train ======================================

def train(
    train_file: str,
    model_name: str,
    output_dir: str,
    dev_file: str = None,                
    max_length: int = 2048,               # CHANGED default
    per_device_train_batch_size: int = 4, # CHANGED default
    gradient_accumulation_steps: int = 8, # CHANGED default
    num_train_epochs: int = 2,
    learning_rate: float = 1e-4,          # CHANGED default
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    seed: int = 42,
    fp16: bool = True,
    bf16: bool = False,
):
    set_seed(seed)
    tok, model = load_tok_and_model(model_name)
    ds_train = load_and_prepare_it_dataset(train_file, tok, max_length, split_name="train")
    ds_eval = None
    if dev_file:  
        ds_eval = load_and_prepare_it_dataset(dev_file, tok, max_length, split_name="dev")

    os.makedirs(output_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    print("Hyperparameters:")
    print(f"  Per device train batch size: {per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Number of train epochs: {num_train_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Warmup ratio: {warmup_ratio}")
    print(f"  LR scheduler type: {lr_scheduler_type}")
    print(f"  Seed: {seed}")
    print(f"  FP16: {fp16}")
    print(f"  BF16: {bf16}")

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
        eval_strategy="epoch",
        save_total_limit=3,
        seed=seed,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        optim="adamw_torch",
        report_to=[],
        load_best_model_at_end=True,  
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    print(f"Done. Model saved to: {output_dir}")

# ============================= CLI ========================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, default="data/it_train.jsonl")  
    ap.add_argument("--dev_file", type=str, default="data/it_dev.jsonl")      
    ap.add_argument("--model", type=str, default="google/gemma-3-270m")       
    ap.add_argument("--output_dir", type=str, default="models/gemma270m-it-v1")

    ap.add_argument("--max_length", type=int, default=2048)                   
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)     
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)     
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=1e-4)              
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
        dev_file=args.dev_file,                
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
