# Ablations & negative results

Several design choices were explored that did **not** improve accuracy. They are documented
here because negative results are useful. The exploratory code lives in
[`src/eval/ablations/`](../src/eval/ablations/).

## BM25 RAG — implemented, and it did not help

A retrieval-augmented pipeline for factual QA was built (BM25 over Wikipedia passages
retrieved via the MediaWiki search endpoint, chunked into paragraphs/sentences and
filtered by normalized score). On TriviaQA, **adding retrieved context hurt accuracy**, and
it got worse as more passages were added, so **retrieval is disabled** in the final system.

| TriviaQA (normalized EM) | No RAG | top-k=2 | top-k=5 |
|---|:---:|:---:|:---:|
| | **0.1554** | 0.1450 | 0.1388 |

*Source: report Table 4.* Likely cause: for a 270M model, imperfectly-matched retrieved
passages inject more distracting noise than useful signal; the model does better relying on
knowledge baked in during SFT. An embedding retriever (all-MiniLM-L6-v2) looked
anecdotally stronger than BM25 on a handful of examples, but project rules disallowed an
external embedding LM at inference time, and an off-the-shelf Gemma encoder performed far
worse than BM25.

## Decoding temperature — negligible effect

Sweeping the decoding temperature had almost no impact on any task.

| Temperature | T=0.3 | T=0.5 | T=0.7 |
|---|:---:|:---:|:---:|
| TriviaQA | 0.1554 | **0.1556** | 0.1534 |
| ARC-C | **0.3308** | 0.3304 | 0.3276 |
| IFEval | **0.4779** | 0.4779 | 0.4760 |

*Source: report Table 3.* A small model like Gemma-3-270M already produces low-entropy,
fairly sharp logits, so temperature scaling barely moves the distribution.

## Few-shot prompting for reasoning — hurt

Adding a 3-shot context to the reasoning expert **decreased** ARC-C from 0.3308 to 0.2928,
so final results use 0-shot. The experts were post-trained on a pretrained (not
instruction-tuned) base and did not adapt well to the long few-shot context or the
`<answer></answer>` tag format.

## Reproducing the ablations

The QA-strategy and Bayesian decoding-hyperparameter search scripts require extra deps:

```bash
pip install -r requirements-dev.txt   # adds scikit-optimize, numpy
python -m src.eval.ablations.qa_strategies --help
python -m src.eval.ablations.hyperparam_search --help
```

*Contributors: the RAG pipelines and much of the ablation/inference-strategy work were done
by teammates — see the attribution note in the [README](../README.md).*
