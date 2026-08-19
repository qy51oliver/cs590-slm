# Evaluation

## Benchmarks and metrics

| Benchmark | What it probes | Metric |
|-----------|----------------|--------|
| TriviaQA | Short-form factual QA | Exact match on normalized text |
| ARC-Challenge (ARC-C) | Multiple-choice reasoning | Accuracy from a single extracted capital letter |
| IFEval | Instruction following | Loose instruction accuracy (via the local `ifeval` package) |

Following the course setup, the TriviaQA `validation` split is used as the test set and the
IFEval `train` split as its test set, because the original test splits ship without
ground-truth answers.

## Methodology

- All models are evaluated **zero-shot**, including the pretrained and instruction-tuned
  baselines, for a fair comparison.
- Reported numbers use **1,000 examples per task, averaged over 5 seeds**.
- The reasoning route uses CoT self-consistency (K=5 majority vote); see
  [ablations.md](ablations.md) for why retrieval and temperature tuning were not adopted.

## Results

| Benchmark | Gemma-3-270M (pretrained) | Gemma-3-270M-IT | **Router-Expert (ours)** |
|-----------|:---:|:---:|:---:|
| TriviaQA  | 0.1229 | 0.1004 | **0.1556** |
| ARC-C     | 0.2365 | 0.2596 | **0.3304** |
| IFEval    | 0.2722 | 0.3726 | **0.4779** |

*Source: report Table 2.* The router-expert system improves over the stronger of the two
baselines on every task.

## Reproduce

Base weights `google/gemma-3-270m` are **gated** — accept the Gemma license on Hugging Face
and `huggingface-cli login` before training. Evaluation of the already-trained SFT
checkpoints needs no gated access.

```bash
pip install -e .            # research package (also installs the vendored IFEval dep set)
pip install -e ./ifeval     # local IFEval evaluator

# Build corpora (writes to data/) and train — see docs/training.md for per-task settings
python scripts/prepare_sft_data.py --if-filter-constraints
python scripts/prepare_router_data.py
python scripts/train_expert.py \
  --train_file data/v3/v3_reasoning_train.jsonl \
  --model google/gemma-3-270m \
  --output_dir models/gemma270m-sft-reasoning \
  --max_length 512 --num_train_epochs 3 --learning_rate 1e-4 --bf16
python scripts/train_router.py \
  --output_dir models/gemma270m-sft-router --bf16

# Evaluate the full system on a task
python scripts/download_eval_data.py
python scripts/evaluate.py --task arc-c --data-size 1000
```

By default `scripts/evaluate.py` points at the published HF checkpoints; pass
`--router_model / --fqa_model / --reas_model / --if_model` to use locally trained ones.
