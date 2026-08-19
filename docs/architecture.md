# Architecture

The system follows a modular **router-expert** design. A lightweight learned router
inspects each query and dispatches it to one of three task-specialized experts, each an
independently fine-tuned copy of `google/gemma-3-270m`.

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

## Routing

The router is a Gemma-3-270M backbone with a **3-way sequence-classification head** over
raw input text, predicting one of `FQA` / `REAS` / `IF`. Because inputs can be long, it
uses **head-tail truncation** (keep a prefix + suffix up to 1024 tokens) so both the
task-type phrasing at the start and any answer choices/constraints at the end survive
truncation — useful for telling multiple-choice questions apart from general instructions.
Unknown labels are treated as errors rather than silently ignored. The serving layer also
surfaces the router's **softmax confidence** for the chosen route.

Implementation: `HFRouterClassifier` in
[`src/inference/pipeline.py`](../src/inference/pipeline.py).

## Experts

Three independent SFT checkpoints, one per route:

| Route | Expert trained on | Inference behaviour |
|-------|-------------------|---------------------|
| `factual_qa` | TriviaQA (`unfiltered.nocontext`, ~87K) | Greedy, short-form answer |
| `reasoning` | ARC-Easy+Challenge, CommonsenseQA, OpenBookQA (~31.5K) | CoT + self-consistency, majority vote over K=5 samples |
| `instruction_following` | Tulu-3 + SmolTalk, constraint-filtered (~50K) | Greedy, free-form response |

See [data.md](data.md) for corpus construction and [training.md](training.md) for the SFT
procedure.

## Inference path

`query → router.predict → expert.generate → answer`

Items are grouped by predicted route so each expert runs a single batched pass. Experts
are **lazy-loaded** on first use (or eagerly warmed at startup in the serving layer via
`SLM_WARM_EXPERTS`). The full pipeline is `OurPipeline` in
[`src/inference/pipeline.py`](../src/inference/pipeline.py); the serving wrapper that adds
routing confidence, latency breakdown, and timeouts is
[`app/inference.py`](../app/inference.py).

The trained checkpoints are published on the Hugging Face Hub, so inference and serving
work without retraining:
[`router`](https://huggingface.co/oliveryql/gemma270m-sft-router) ·
[`fqa`](https://huggingface.co/oliveryql/gemma270m-sft-fqa) ·
[`reasoning`](https://huggingface.co/oliveryql/gemma270m-sft-reasoning) ·
[`if`](https://huggingface.co/oliveryql/gemma270m-sft-if).
