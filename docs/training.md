# Training

Each expert is a **causal LM fine-tuned with a prompt-masked SFT objective**: prompt tokens
are masked (`-100`) and cross-entropy is applied only to the target tokens, with an EOS
appended to the target. Crucially, the model's **chat template is applied during training**
so the training-time input format matches inference-time formatting.

- Expert SFT: [`src/train/expert.py`](../src/train/expert.py) (CLI: `scripts/train_expert.py`)
- Router training: [`src/train/router.py`](../src/train/router.py) (CLI: `scripts/train_router.py`)

## Expert SFT hyperparameters

All three experts share this configuration:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4, cosine schedule, warmup ratio 0.03 |
| Weight decay | 0.01 |
| Epochs | 3 |
| Batch size | 16 |
| Precision | BF16 |

**Per-task maximum sequence lengths:** 768 (factual QA), 512 (reasoning), 2048
(instruction following). Instruction-following additionally applies a hard total-token cap
to avoid over-weighting long conversations under a fixed budget.

**Checkpoint selection.** Each corpus is split 90/10; the checkpoint with the **lowest
validation loss** is kept as the final expert.

### Prompt/target formats

- **Factual QA:** `Answer the question concisely.\nQuestion: <q>\nAnswer:` → first
  normalized answer alias.
- **Multiple-choice reasoning:** an instruction enforcing a single-letter A–D output → the
  gold option letter.
- **Instruction following:** the instruction text itself as the user turn → the gold
  response.

## Router training hyperparameters

3-way sequence-classification head, fixed label mapping `{FQA, REAS, IF}`, max length 1024
with head-tail truncation for overlength inputs.

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4, cosine schedule, warmup ratio 0.06 |
| Weight decay | 0.01 |
| Epochs | 3 |
| Batch size | 16 |
| Precision | BF16 |

*Source: project report §3.3–3.4 and Appendix A.1.*
