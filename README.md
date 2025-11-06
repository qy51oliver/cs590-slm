## CS590 Final Project: Fine-tuning a Small Language Model

This repository contains the starter code and tooling to evaluate a small language model (tuned on `google/gemma-3-270m`) on three tasks that probe different capabilities:

- **TriviaQA**: factual QA (short-form factual answers)
- **ARC-Challenge (ARC-C)**: multiple-choice reasoning
- **IFEval**: instruction following

You will train your own model, implement your own inference pipeline and submit predictions to the Gradescope autograder.

### What you will build
- A model fine-tuned based on Gemma-3-270m 
- A submission archive (`submissions.zip`) with predictions for the three tasks.
- [Optional] A custom pipeline in `OurPipeline` that handle prompt formatting, batching and optional routing.
> The provided pipeline is just an example, you don’t need to follow this class exactly. You’re free to implement it in any way that feels most natural to you, as long as your code takes in questions and produces predictions in the same format as shown in our examples.


## Repository structure
- `pipelines.py`: model loading, prompt formatting, batching, and routing.
  - `BasePipeline`: base inference which handle different tasks with shared logic.
  - `RouterPipeline`: example router that handle different tasks with different logic. (Illustrative only; you must implement your own router if you choose this approach instead of using task_type.)
  - `OurPipeline`: where you implement your strategy. This is the primary TODO. You don't need to follow the structure exactly; feel free to implement your own design. Just make sure it takes in questions and produces text outputs. 
- `generate.py`: reads a JSONL of inputs and produces predictions JSONL.
- `score.py`: computes task-specific scores (TriviaQA F1/EM, ARC-C accuracy, IFEval via the local `ifeval` package).
- `eval.py`: one-shot pipeline: generate → score for a single task/dataset.
- `submit.py`: runs multiple stochastic seeds, aggregates metrics, writes three prediction JSONLs, and packages `submissions.zip` for Gradescope.
- `download_eval_data.py`: downloads and normalizes the three evaluation datasets from Hugging Face into `./data`.
- `ifeval/`: local package providing the IFEval evaluator and language logic.


## Setup

### Create environment and install dependencies
```bash
# Install top-level package and dependencies (transformers, torch, datasets, tqdm)
pip install -e .

# Install IFEval local package and its extras
pip install -e ./ifeval
```


## Download evaluation datasets
This will create normalized JSONL files under `./data`.
```bash
python download_eval_data.py 
```
Generated files (defaults):
- `data/triviaqa_test.jsonl`
- `data/arc_c_test.jsonl`
- `data/ifeval_test.jsonl`

> Note: we use train split of ifeval as its test split and we use dev split of triviaqa as test split as the original test splits do not have ground-truth answers.

## Quickstart: evaluate a model
Run end-to-end generate → score on a task with a chosen model.
```bash
python eval.py \
  --task arc-c \
  --model google/gemma-3-270m-it \
  --data-size 1000 \
  --batch-size 512 \
  --max-new-tokens 512
```
Outputs:
- Predictions: `outputs/<dataset>_preds.jsonl`
- A JSON summary is printed to stdout (and contains absolute paths to the outputs).

Supported tasks for `--task`: `triviaqa`, `arc-c`, `ifeval`.

> Note: The gemma-3-270m-it model builds on gemma-3-270m by fine-tuning it on instruction-following data. Try fine-tuning your own model from gemma-3-270m and see if you can reach—or even beat—the performance of gemma-3-270m-it on the three tasks.


## Generate only (without scoring)
```bash
python generate.py \
  --data_file data/triviaqa_test.jsonl \
  --model google/gemma-3-270m-it \
  --data-size 1000 \
  --batch-size 512 \
  --max-new-tokens 256
```
The output will be written under `outputs/` as `<data_file>_preds.jsonl`.


## Score only
Given a JSONL that contains the original inputs merged with a `"prediction"` field (the format produced by `generate.py` and `eval.py`):
```bash
python score.py \
  --task ifeval \
  --preds outputs/ifeval_test_preds.jsonl
```


## Implementing your pipeline (the main TODO)
Open `pipelines.py` and implement your strategy in `OurPipeline`.

### Where to customize
Feel free to modify the class anywhere, just make sure the inputs and outputs of the pipeline remain compatible with the rest of the codebase. Examples of common customization points:
- **Prompt construction**: override `ProcessLogic.preprocess` (or use the provided `FactualQAProcessor`, `ReasoningProcessor`, `InstructionFollowingProcessor`). 
- **Post-processing**: override `ProcessLogic.postprocess` to clean model outputs into the desired form.
- **Routing (optional)**: if you adopt a router strategy, you must train/learn a router. The example `RouterPipeline._router` uses `task_type` only for illustration and must not be used as-is for a real router.
- **Decoding parameters**: adjust `max_new_tokens`, sampling, temperature, etc. via CLI or by overriding `run`.


## Reproducible multi-run submission
After you have your model and pipeline ready, you can create your submission to upload to Gradescope to get part of the project grade.
You will need to run five seeds per task to obtain averaged metrics. Additionally, to save computation, we limit the data size to 1000 examples per task.

Use `submit.py` to generate the three required JSONLs (one per task), aggregate metrics across seeds, and build `submissions.zip`.
```bash
python submit.py \
  --model <your_own_model> \
  --pipeline our \
  --seeds 1,2,3,4,5 \
  --data_size 1000 
```

Upload resulted `submissions.zip` to the Gradescope autograder. Ensure your JSONL structure matches the examples produced by this script.


