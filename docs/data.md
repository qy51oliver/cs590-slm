# Data pipeline

All training data comes exclusively from the **train** splits of six public datasets,
unified into a single JSONL schema capturing `task_type`, `question`, and task-dependent
`answers` (or `answerKey` for multiple-choice). ~168K SFT examples total across the three
expert corpora.

Built by [`src/data/prepare_sft.py`](../src/data/prepare_sft.py)
(CLI: `scripts/prepare_sft_data.py`).

| Expert corpus | Sources | Size | Key normalization |
|---------------|---------|------|-------------------|
| Factual QA | TriviaQA (`unfiltered.nocontext`) | ~87K | Answers stored as an alias list; first alias is the SFT target |
| Reasoning | ARC-Easy + ARC-Challenge, CommonsenseQA, OpenBookQA | ~31.5K | See below |
| Instruction following | Tulu-3 instruction-following + SmolTalk | ~50K (~30K + ~20K) | See below |

## Reasoning corpus normalization

- **CommonsenseQA 5→4 choices.** CommonsenseQA ships 5 answer choices; each example is
  subsampled to four by keeping the correct choice, sampling three distractors, and
  re-lettering to A–D so the format matches ARC/OpenBookQA.
- **ARC answer-choice shuffles.** To reduce choice-order bias, each ARC question is
  augmented with label-preserving permutations of its options (the correct-answer label is
  updated accordingly). This expands the reasoning corpus to ~31.5K.

## Instruction-following corpus normalization

- **Multi-turn chat flattening.** Both Tulu-3 and SmolTalk can be multi-turn message
  lists; each example is flattened into a single instruction text (concatenated
  non-assistant messages) and a single target response (the assistant message). Role
  markers ("User:", "Assistant:") are stripped, leaving plain instruction/response pairs.
- **Constraint filtering for IFEval.** Because IFEval measures satisfaction of explicit
  constraints, a regex heuristic keeps only training examples that contain
  constraint-like language ("exactly", "at least", "must include", "uppercase", "avoid").
  This focuses training on the kinds of examples IFEval rewards. Result: ~30K from Tulu-3
  and ~20K from SmolTalk.

## Router dataset

Built by [`src/data/prepare_router.py`](../src/data/prepare_router.py)
(CLI: `scripts/prepare_router_data.py`) directly from the three expert corpora: examples
are relabeled `{FQA, REAS, IF}` and **balance-sampled to 15K per class (45K total)**, plus
a small mined pool of "hard" instruction-following examples (code/math-like) to sharpen the
IF/REAS boundary. Split into train/validation before training.

*Sources: project report §3.2 (expert corpora) and §3.4 (router dataset).*
