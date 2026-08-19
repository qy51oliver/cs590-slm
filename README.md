# Router-Expert SLM: Task-Specialized Fine-Tuning of Gemma-3-270M

A modular **router-expert** system that adapts the 270M-parameter `google/gemma-3-270m`
small language model to three different task types. A lightweight, learned **3-way
router classifier** inspects each incoming query and dispatches it to one of three
**task-specialized experts** — factual QA, multiple-choice reasoning, or instruction
following — each independently fine-tuned with supervised fine-tuning (SFT) on its own
curated corpus. On the three held-out benchmarks the router-expert system beats both the
pretrained and the official instruction-tuned Gemma-3-270M baselines. This repo contains
the full research pipeline (data prep → SFT → router training → evaluation) **and** a
production-style FastAPI serving layer with Docker packaging and a latency benchmark.

> Model size: **270M parameters per expert.** These are small models — the value here is
> the routing architecture, the data/training pipeline, and the deployment story, not
> frontier answer quality. All accuracy numbers below come from the project report; all
> latency numbers come from the benchmark in this repo (see [sources](#where-the-numbers-come-from)).

---

## Results

Zero-shot accuracy on the three evaluation benchmarks (1,000 examples per task, averaged
over 5 seeds). **Router-Expert is our system.** Higher is better.

| Benchmark | Metric | Gemma-3-270M (pretrained) | Gemma-3-270M-IT (instruction-tuned) | **Router-Expert (ours)** |
|-----------|--------|:---:|:---:|:---:|
| TriviaQA  | normalized EM        | 0.1229 | 0.1004 | **0.1556** |
| ARC-C     | accuracy             | 0.2365 | 0.2596 | **0.3304** |
| IFEval    | loose instr. accuracy| 0.2722 | 0.3726 | **0.4779** |

*Source: Table 2 of the project report. The pretrained and instruction-tuned baselines
are evaluated zero-shot for a fair comparison.*

The router-expert system improves over the **stronger** of the two baselines on every
task: +0.033 TriviaQA (vs. pretrained), +0.071 ARC-C, and +0.105 IFEval (vs. IT).

---

## Architecture

```
                         ┌─────────────────────────────┐
        query  ─────────▶│   Router (Gemma-3-270M       │
                         │   sequence classifier)       │
                         │   → {FQA, REAS, IF} + conf    │
                         └──────────────┬──────────────┘
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
          ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
          │  Factual-QA      │ │  Reasoning       │ │  Instruction-    │
          │  expert (SFT)    │ │  expert (SFT)    │ │  Following expert│
          │                  │ │  CoT + self-     │ │  (SFT)           │
          │                  │ │  consistency K=5 │ │                  │
          └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
                   └────────────────────┼────────────────────┘
                                        ▼
                                     answer
```

**Routing.** The router is a Gemma-3-270M backbone with a 3-way sequence-classification
head over raw input text, predicting one of `FQA` / `REAS` / `IF`. Because inputs can be
long, it uses a **head-tail truncation** strategy (keep a prefix + suffix up to 1024
tokens) so both the task-type phrasing at the start and any answer choices/constraints at
the end survive truncation. Unknown labels are treated as errors, not silently ignored.
The serving layer also exposes the router's **softmax confidence** for the chosen route.

**Experts.** Three independent SFT checkpoints, one per route:

| Route | Expert trained on | Inference behaviour |
|-------|-------------------|---------------------|
| `factual_qa` | TriviaQA (`unfiltered.nocontext`, ~87K) | Greedy, short-form answer |
| `reasoning` | ARC-Easy+Challenge, CommonsenseQA, OpenBookQA (~31.5K) | CoT + self-consistency, majority vote over K=5 samples |
| `instruction_following` | Tulu-3 + SmolTalk, constraint-filtered (~50K) | Greedy, free-form response |

**Inference path.** `query → router.predict → expert.generate → answer`. Items are grouped
by predicted route so each expert runs a single batched pass; experts are lazy-loaded on
first use (or eagerly warmed at startup in the serving layer). See
[`pipelines_inference.py`](pipelines_inference.py) (`OurPipeline`).

The trained checkpoints are published on the Hugging Face Hub, so inference and serving
work without retraining:
[`router`](https://huggingface.co/oliveryql/gemma270m-sft-router) ·
[`fqa`](https://huggingface.co/oliveryql/gemma270m-sft-fqa) ·
[`reasoning`](https://huggingface.co/oliveryql/gemma270m-sft-reasoning) ·
[`if`](https://huggingface.co/oliveryql/gemma270m-sft-if).

---

## Serving layer

A FastAPI service ([`app/`](app/)) wraps the router-expert system for online inference:
models load **once at startup**, requests are typed with Pydantic, every request is
logged as structured JSON with a latency breakdown, and failure modes (model-load failure,
malformed input, generation timeout) return typed error responses.

### Endpoints

| Method | Path        | Purpose |
|--------|-------------|---------|
| `POST` | `/generate` | Route a query, run the selected expert, return the answer + routing metadata + latency. |
| `GET`  | `/health`   | Liveness/readiness: router loaded, which experts are loaded, device, version. |

`GET /docs` serves the auto-generated OpenAPI UI.

### Request / response

```bash
curl -s http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"query": "List three benefits of regular exercise. Use exactly three bullet points.",
       "max_new_tokens": 128}'
```

```json
{
  "answer": "- Improved cardiovascular health by increasing blood flow to the muscles...\n- Enhanced mental and physical well-being by reducing stress...\n- ...",
  "route": "instruction_following",
  "router_confidence": 1.0,
  "latency_ms": 1202.0,
  "router_latency_ms": 18.7,
  "generation_latency_ms": 1183.3,
  "request_id": "aff87d0bc55d"
}
```

`GET /health`:

```json
{
  "status": "ok",
  "router_loaded": true,
  "experts_loaded": ["factual_qa", "reasoning", "instruction_following"],
  "device": "mps:0",
  "version": "0.1.0"
}
```

The service is configured entirely through environment variables (prefix `SLM_`): model
ids/paths (`SLM_ROUTER_MODEL`, `SLM_FQA_MODEL`, …), decoding defaults (`SLM_MAX_NEW_TOKENS`,
`SLM_REASONING_SC_K`, `SLM_TEMPERATURE`, `SLM_TOP_P`), `SLM_WARM_EXPERTS`,
`SLM_GENERATION_TIMEOUT_S`, and logging (`SLM_LOG_LEVEL`, `SLM_LOG_JSON`). See
[`app/config.py`](app/config.py) for the full list and defaults.

### Run it locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# first start downloads the 4 checkpoints (~2 GB) from the HF Hub
```

### Run it in Docker

```bash
docker compose up --build       # builds the image and starts the service on :8000
# or, without compose:
docker build -t router-expert-slm .
docker run -p 8000:8000 -v hf-cache:/home/appuser/.cache/huggingface router-expert-slm
```

The [`Dockerfile`](Dockerfile) is multi-stage (build venv → slim runtime), runs as a
non-root user, installs **CPU-only** Torch to keep the image small, and ships a
`HEALTHCHECK`. [`docker-compose.yml`](docker-compose.yml) mounts a named volume for the HF
cache so weights download only once.

### Measured latency

Benchmark harness: [`benchmarks/benchmark.py`](benchmarks/benchmark.py) sends a fixed set
of queries spanning all three routes and reports server-side latency percentiles and
throughput, overall and per route.

**Environment:** Apple M4 (10 cores, 24 GB), macOS 15.7.3, Python 3.12, PyTorch 2.9.1 on
the **MPS** backend, float32. All four models warmed at startup, sequential requests
(`concurrency=1`), 26 timed requests.

| Route (expert) | n | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|----------------|---|:---:|:---:|:---:|:---:|
| **overall** | 26 | 307.1 | 2618.8 | 3079.8 | 600.9 |
| factual_qa | 10 | 114.3 | 200.1 | 215.9 | 128.4 |
| reasoning *(CoT-SC, K=5)* | 8 | 308.3 | 316.1 | 316.7 | 309.0 |
| instruction_following | 8 | 1086.8 | 3078.6 | 3081.8 | 1483.3 |

Throughput: **1.66 req/s** sequential (wall time 15.6 s for 26 requests). Latency is
dominated by output length: factual answers are short; instruction-following generates up
to `max_new_tokens` of free text (widest tail); the reasoning route runs K=5 sampled CoT
passes but stops early on short letter answers. Generation is serialized behind a
single-worker executor (one accelerator), so throughput scales with sequence length rather
than request concurrency.

Reproduce:

```bash
uvicorn app.main:app --port 8000 &          # terminal 1
python benchmarks/benchmark.py --url http://localhost:8000 --repeats 2   # terminal 2
```

---

## Data

All training data comes exclusively from the **train** splits of six public datasets,
unified into a single JSONL schema capturing `task_type`, `question`, and task-dependent
`answers` (or `answerKey` for multiple-choice). ~168K SFT examples total across the three
expert corpora.

| Expert corpus | Sources | Size | Key normalization |
|---------------|---------|------|-------------------|
| Factual QA | TriviaQA (`unfiltered.nocontext`) | ~87K | Answers stored as an alias list; first alias is the SFT target |
| Reasoning | ARC-Easy + ARC-Challenge, CommonsenseQA, OpenBookQA | ~31.5K | **CommonsenseQA 5→4 choices** (keep the correct option, sample 3 distractors, re-letter A–D); ARC augmented with label-preserving answer-choice shuffles for order-robustness |
| Instruction following | Tulu-3 instruction-following + SmolTalk | ~50K (~30K + ~20K) | **Multi-turn chats flattened** to a single instruction/response pair; role markers stripped; **regex constraint filtering** keeps examples with IFEval-style constraints ("exactly", "at least", "must include", "uppercase", "avoid") |

**Router dataset.** Built directly from the three expert corpora: examples are relabeled
`{FQA, REAS, IF}` and **balance-sampled to 15K per class (45K total)**, plus a small mined
pool of "hard" instruction-following examples (code/math-like), for routing that mirrors
the experts' domains. See [`prepare_data.py`](prepare_data.py) and
[`prepare_router_data.py`](prepare_router_data.py).

---

## Training

Each expert is a **causal LM fine-tuned with a prompt-masked SFT objective**: prompt tokens
are masked (`-100`) and cross-entropy is applied only to the target tokens, with an EOS
appended to the target. Crucially, the model's **chat template is applied during training**
so the training-time input format matches inference-time formatting.

**Expert SFT** (all three share): AdamW, LR `1e-4`, cosine schedule, warmup ratio `0.03`,
weight decay `0.01`, 3 epochs, batch size 16, **BF16**. Per-task maximum sequence lengths:
**768 (factual QA), 512 (reasoning), 2048 (instruction following)**; instruction-following
additionally applies a hard total-token cap to avoid over-weighting long conversations.
Each corpus is split 90/10 and the **checkpoint with the lowest validation loss** is kept.

**Router training.** 3-way sequence-classification head, fixed label mapping
`{FQA, REAS, IF}`, max length 1024 with head-tail truncation. AdamW, LR `1e-4`, cosine,
warmup ratio `0.06`, weight decay `0.01`, 3 epochs, batch size 16, BF16.

*Source: report Appendix A.1 / §3.3–3.4.*

---

## Inference details

- **CoT self-consistency (reasoning route).** For multiple-choice questions the reasoning
  expert is prompted to reason step-by-step, sampled **K=5** times with independent
  rollouts; the final answer letter is chosen by **majority vote**, reducing variance from
  any single reasoning path.

- **BM25 RAG — implemented, and honestly, it did not help.** A retrieval-augmented pipeline
  for factual QA was built (BM25 over Wikipedia passages retrieved via the MediaWiki search
  endpoint, chunked into paragraphs/sentences, normalized-score filtered — see
  [`rag_smoke_test.py`](rag_smoke_test.py)). On TriviaQA, adding retrieved context **hurt**
  accuracy and got worse as more passages were added, so **retrieval is disabled** in the
  final system:

  | TriviaQA (normalized EM) | No RAG | top-k=2 | top-k=5 |
  |---|:---:|:---:|:---:|
  | | **0.1554** | 0.1450 | 0.1388 |

  *Source: report Table 4.* The likely cause: for a 270M model, imperfectly-matched
  retrieved passages inject more distracting noise than useful signal, and the model does
  better relying on knowledge baked in during SFT. Temperature tuning (0.3/0.5/0.7) was
  also ablated and had negligible effect (report Table 3).

- **Metrics.** TriviaQA = exact match on normalized text; ARC-C = accuracy from a single
  extracted capital letter; IFEval = loose instruction accuracy via the local `ifeval`
  package.

---

## Reproduce

Base weights `google/gemma-3-270m` are **gated** — accept the Gemma license on Hugging Face
and `huggingface-cli login` before training. (Serving/eval use the already-trained SFT
checkpoints and need no gated access.)

```bash
# 0. Install
pip install -e .            # research package (torch, transformers, datasets, accelerate, tqdm)
pip install -e ./ifeval     # local IFEval evaluator
pip install -r requirements.txt   # adds the serving deps

# 1. Build the three expert corpora + the router dataset (writes to data/)
python prepare_data.py --if-filter-constraints
python prepare_router_data.py

# 2. Train an expert (example: reasoning). Repeat per task with its own file/seq-length.
python finetune_expert.py \
  --train_file data/v3/v3_reasoning_train.jsonl \
  --model google/gemma-3-270m \
  --output_dir models/gemma270m-sft-reasoning \
  --max_length 512 --num_train_epochs 3 --learning_rate 1e-4 --bf16

# 3. Train the router
python finetune_router.py \
  --train_file data/router_cls/router_cls_train.jsonl \
  --val_file   data/router_cls/router_cls_val.jsonl \
  --model google/gemma-3-270m \
  --output_dir models/gemma270m-sft-router --bf16

# 4. Download the eval sets and evaluate the full system on a task
python download_eval_data.py
python eval.py --task arc-c --data-size 1000
```

By default `eval.py` (and the serving layer) point at the published HF checkpoints; pass
`--router_model / --fqa_model / --reas_model / --if_model` (or the `SLM_*` env vars) to use
locally trained ones.

---

## Repository structure

```
app/                       FastAPI serving layer
  main.py                  Endpoints (/generate, /health), lifespan model load, error handling
  inference.py             RouterExpertService: routing + dispatch + timeout wrapper
  config.py                Env-var configuration (pydantic-settings)
  schemas.py               Pydantic request/response models
  logging_config.py        Structured JSON logging
benchmarks/
  benchmark.py             Latency (p50/p95/p99) + throughput harness
  queries.jsonl            Fixed multi-route query set used by the benchmark
pipelines_inference.py     Core inference: router classifier, experts, CoT-SC (OurPipeline)
prepare_data.py            Build the three expert SFT corpora (unified JSONL)
prepare_router_data.py     Build the balanced 3-way router classification dataset
finetune_expert.py         Prompt-masked SFT for a single expert
finetune_router.py         Train the 3-way router classification head
download_eval_data.py      Download + normalize TriviaQA / ARC-C / IFEval eval sets
eval.py / generate.py / score.py   End-to-end evaluate / generate-only / scoring
rag_smoke_test.py          BM25 Wikipedia RAG prototype (disabled in the final system)
pipelines_qa_strategies.py / eval_qa_strategies.py / eval_ablation.py   Inference-strategy ablations
ifeval/                    Vendored IFEval evaluator package
Dockerfile / docker-compose.yml / requirements*.txt
```

Model weights, datasets, and run outputs are git-ignored and pulled/generated on demand.

---

## Where the numbers come from

- **Accuracy tables (Results, RAG):** verbatim from the project report — Table 2 (main
  results), Table 4 (RAG ablation), Table 3 (temperature).
- **Data sizes, training hyperparameters, sequence lengths, K=5:** report §3 and Appendix A.1.
- **Serving latency table + throughput:** produced by running
  `benchmarks/benchmark.py` against the live service on the machine described above.
  Re-running regenerates `benchmarks/results/latest.json`.

---

## Team & attribution

This was a group project for Duke COMPSCI 590. This repository packages the work I
(**Oliver You**) led: the supervised fine-tuning data pipeline, the final task-specific
training corpora, the expert SFT training scripts, and the design/training of the 3-way
router used in the final router-expert system. The BM25/embedding RAG pipeline, inference-
strategy and ablation studies, and report analysis were contributed by teammates
(Leonardo Biral, Ziqi Zhou, Jiajun Cheng). The FastAPI serving layer, Docker packaging, and
benchmark in this repo were added afterward to demonstrate a production inference path.
