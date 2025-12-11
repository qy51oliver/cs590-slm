from typing import Any, Dict, List, Optional
from collections import Counter
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm

# ---- Router classifier (HF sequence classification) ----
_LABEL2ROUTE = {"FQA": "factual_qa", "REAS": "reasoning", "IF": "instruction_following"}
_ALLOWED_ROUTES = set(_LABEL2ROUTE.values())


class HFRouterClassifier:
    """
    Uses a Gemma-3-270M sequence-classification head trained on labels: FQA / REAS / IF.
    Predicts route keys: 'factual_qa' | 'reasoning' | 'instruction_following'.
    """

    def __init__(self, model_name: str, max_length: int = 1024, head_ratio: float = 0.7):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Classifier training used right padding
        self.tok.padding_side = "right"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()
        self.max_length = int(max_length)
        self.head_ratio = float(head_ratio)

    def _head_tail(self, ids: List[int]) -> List[int]:
        # Keep head portion + tail portion if exceeding max_length
        if len(ids) <= self.max_length:
            return ids
        head = int(self.max_length * self.head_ratio)
        tail = max(1, self.max_length - head)
        return ids[:head] + ids[-tail:]

    @torch.no_grad()
    def predict_routes(self, texts: List[str]) -> List[str]:
        # texts: list of raw prompts/questions for classification
        encs: List[Dict[str, Any]] = []
        for t in texts:
            ids = self.tok(t.strip(), add_special_tokens=True, truncation=False)["input_ids"]
            ids = self._head_tail(ids)
            encs.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        batch = self.tok.pad(encs, padding=True, return_tensors="pt").to(self.model.device)

        logits = self.model(**batch).logits  # shape [B, 3]
        pred_ids = logits.argmax(dim=-1).tolist()
        # id2label exists from training; fallback assumes FQA/REAS/IF in order
        id2label = getattr(self.model.config, "id2label", {0: "FQA", 1: "REAS", 2: "IF"})
        labels = [id2label[int(i)] for i in pred_ids]
        routes: List[str] = []
        for lab in labels:
            if lab not in _LABEL2ROUTE:
                raise ValueError(f"Router produced unknown label: {lab}")
            routes.append(_LABEL2ROUTE[lab])
        return routes


# ---------------- helpers ----------------
def apply_instruct_template(prompt: Any, tokenizer: PreTrainedTokenizerBase) -> str:
    """
    Apply chat_template when available; otherwise construct a simple textual format.
    prompt: can be a list of {"role","content"} messages or a raw string.
    """
    has_chat = tokenizer and hasattr(tokenizer, "chat_template") and tokenizer.chat_template
    if has_chat:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(prompt, list):
        lines: List[str] = []
        for m in prompt:
            role = m.get("role", "user")
            prefix = "User" if role == "user" else ("Assistant" if role == "assistant" else role.capitalize())
            lines.append(f"{prefix}: {m.get('content', '')}")
        return "\n".join(lines)
    return str(prompt)


def _strip_code_fences(text: str) -> str:
    # Remove code-block fences at start of lines
    text = re.sub(r"^\s*```.*$", "", text, flags=re.MULTILINE)
    return text.strip()


@torch.no_grad()
def _batched_generate_hf(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    *,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 16,
) -> List[str]:
    """
    Generic batched generation wrapper.
    Prompts are already fully formatted; only decoding is handled here.
    """
    outs: List[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Only return newly generated tokens
        dec = tokenizer.batch_decode(gen[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
        outs.extend([d.strip() for d in dec])
    return outs


# ---------------- logic-only processors ----------------
class ProcessLogic:
    """
    Logic layer defining preprocess/postprocess.
    Independent of the underlying model architecture.
    """

    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        return [apply_instruct_template(str(it.get("prompt", "")), tokenizer) for it in items]

    def postprocess(self, outputs: List[str], items: List[Dict[str, Any]]) -> List[str]:
        return [_strip_code_fences(t).strip() for t in outputs]


class FactualQAProcessor(ProcessLogic):
    """
    TriviaQA / factual_qa:
    - "Answer the question concisely."
    - Optional few-shot controlled externally (default off).
    """

    def __init__(self, *, few_shot: int = 0):
        self.few_shot = max(0, min(few_shot, 5))

    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        prompts: List[str] = []
        for it in items:
            q = it.get("question") or it.get("query") or ""
            if self.few_shot <= 0:
                messages = [
                    {
                        "role": "user",
                        "content": f"Answer the question concisely.\nQuestion: {q}\nAnswer:",
                    }
                ]
                prompts.append(apply_instruct_template(messages, tokenizer))
            else:
                examples = it.get("few_shot_examples", [])[: self.few_shot]
                parts: List[str] = ["Answer the question concisely."]
                for ex in examples:
                    ex_q = ex.get("question", "")
                    ex_a = ex.get("answer", "")
                    parts.append(f"Question: {ex_q}\nAnswer: {ex_a}")
                parts.append(f"Question: {q}\nAnswer:")
                messages = [{"role": "user", "content": "\n".join(parts)}]
                prompts.append(apply_instruct_template(messages, tokenizer))
        return prompts

    def postprocess(self, outputs: List[str], items: List[Dict[str, Any]]) -> List[str]:
        cleaned = super().postprocess(outputs, items)
        # TriviaQA outputs: take only the first line
        return [c.split("\n")[0].strip() for c in cleaned]


class ReasoningProcessor(ProcessLogic):
    """
    ARC-C / reasoning:
    - CoT-style prompt requiring final answer format 'Final answer: X'.
    """

    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        prompts: List[str] = []
        for it in items:
            question = it.get("question", "")
            content = (
                "You will be given a multiple-choice question with options A–D.\n"
                "First think through the problem step-by-step.\n"
                "At the end, output ONLY one line in the format:\n"
                "Final answer: X\n"
                "where X is exactly one capital letter from {A, B, C, D}.\n\n"
                "Question:\n"
                f"{question.strip()}\n\n"
                "Final answer:"
            )
            prompts.append(
                apply_instruct_template(
                    [{"role": "user", "content": content}],
                    tokenizer,
                )
            )
        return prompts


class InstructionFollowingProcessor(ProcessLogic):
    """
    IFEval / instruction_following:
    - No extra meta-instruction; directly use original prompt.
    """

    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        prompts: List[str] = []
        for it in items:
            txt = it.get("question") or it.get("query") or it.get("prompt") or ""
            prompts.append(apply_instruct_template(str(txt), tokenizer))
        return prompts


# ---------------- base inference pipeline ----------------
class BasePipeline:
    """
    Single-model inference wrapper using HF CausalLM.
    Uses a supplied ProcessLogic for one generation pass.
    """

    def __init__(self, model_name: str):
        print(f"[BasePipeline] Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Left padding recommended for decoder-only models
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _generate(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        batch_size: int,
    ) -> List[str]:
        # Prompts are already formatted (chat_template applied upstream)
        return _batched_generate_hf(
            self.model,
            self.tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )

    def run(
        self,
        items: List[Dict[str, Any]],
        *,
        logic: Optional[ProcessLogic] = None,
        batch_size: int = 16,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if logic is None:
            logic = ProcessLogic()
        prompts = logic.preprocess(items, self.tokenizer)
        raw = self._generate(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )
        return logic.postprocess(raw, items)


# ---------------- simple RouterPipeline (task_type-based routing) ----------------
class RouterPipeline(BasePipeline):
    """
    Simple router based on item['task_type'].
    Mainly for sanity checks or baseline comparisons.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def _router(self, item: Dict[str, Any]) -> str:
        task_type = item.get("task_type", "factual_qa").lower()
        task_type_map = {
            "triviaqa": "factual_qa",
            "arc-c": "reasoning",
            "ifeval": "instruction_following",
        }
        return task_type_map.get(task_type, "factual_qa")

    def run_with_router(
        self,
        items: List[Dict[str, Any]],
        router_fn,
        logic_map: Optional[Dict[str, ProcessLogic]] = None,
        batch_size: int = 16,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        # Group items by route key
        groups: Dict[str, List[int]] = {}
        for idx, it in enumerate(items):
            key = router_fn(it)
            groups.setdefault(key, []).append(idx)

        # Default processors
        if logic_map is None:
            logic_map = {
                "factual_qa": FactualQAProcessor(),
                "reasoning": ReasoningProcessor(),
                "instruction_following": InstructionFollowingProcessor(),
            }

        outputs: List[Optional[str]] = [None] * len(items)
        for key, idxs in groups.items():
            sub_items = [items[i] for i in idxs]
            logic = logic_map.get(key, FactualQAProcessor())
            preds = super().run(
                sub_items,
                logic=logic,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            for i, p in zip(idxs, preds):
                outputs[i] = p
        return [o if o is not None else "" for o in outputs]

    def run(
        self,
        items: List[Dict[str, Any]],
        logic: Optional[ProcessLogic] = None,
        **gen_kwargs: Any,
    ) -> List[str]:
        return self.run_with_router(items, self._router, **gen_kwargs)


# ---------------- OurPipeline: classifier router + experts + reasoning SC ----------------
def _text_for_router(item):
    # Extract text for routing based on available fields
    for k in ("question", "prompt", "query", "instruction", "input"):
        if k in item and item[k] is not None:
            return str(item[k])
    return ""


def _extract_choice_letter(text: str) -> Optional[str]:
    """
    Strict parsing of multiple-choice output to {A,B,C,D}.
    Prefer 'Final answer: X' pattern, then last-line single letter, then any letter in text.
    """
    m = re.search(r"Final answer:\s*([ABCD])", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        m2 = re.search(r"\b([ABCD])\b", last.upper())
        if m2:
            return m2.group(1).upper()

    found = re.findall(r"[ABCD]", text.upper())
    if found:
        return found[-1].upper()

    return None


class OurPipeline:
    """
    Full router pipeline:
      - Classifier router → route each item
      - Three experts (lazy-loaded)
      - Reasoning route uses CoT + self-consistency (SC)
    """

    def __init__(
        self,
        router_model: Optional[str] = None,
        experts: Optional[Dict[str, str]] = None,
        router_max_len: int = 1024,
        evict_after_route: bool = False,
        reasoning_sc_K: int = 5,
    ):
        # Router classifier model
        if router_model is None:
            router_model = "oliveryql/gemma270m-sft-router"

        # Expert models
        if experts is None:
            experts = {
                "factual_qa": "oliveryql/gemma270m-sft-fqa",
                "reasoning": "oliveryql/gemma270m-sft-reasoning",
                "instruction_following": "oliveryql/gemma270m-sft-if",
            }

        missing = _ALLOWED_ROUTES - set(experts.keys())
        if missing:
            raise ValueError(f"experts missing routes: {sorted(missing)}")

        # Router
        self.router = HFRouterClassifier(router_model, max_length=router_max_len)

        # Expert storage (lazy load)
        self._expert_names: Dict[str, str] = dict(experts)
        self._expert_pipes: Dict[str, Optional[BasePipeline]] = {k: None for k in experts.keys()}

        # One processor per route
        self.logic_map: Dict[str, ProcessLogic] = {
            "factual_qa": FactualQAProcessor(),
            "reasoning": ReasoningProcessor(),
            "instruction_following": InstructionFollowingProcessor(),
        }

        self.evict_after_route = bool(evict_after_route)
        self.reasoning_sc_K = max(1, int(reasoning_sc_K))

    def _get_expert(self, route_key: str) -> BasePipeline:
        # Lazy load expert model
        pipe = self._expert_pipes.get(route_key)
        if pipe is not None:
            return pipe
        model_name = self._expert_names[route_key]
        pipe = BasePipeline(model_name)
        self._expert_pipes[route_key] = pipe
        return pipe

    def _evict_expert(self, route_key: str) -> None:
        # Free model memory (optional)
        pipe = self._expert_pipes.get(route_key)
        if pipe is None:
            return
        try:
            if hasattr(pipe, "model"):
                del pipe.model
            if hasattr(pipe, "tokenizer"):
                del pipe.tokenizer
        except Exception:
            pass
        self._expert_pipes[route_key] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()

    def _run_reasoning_with_sc(
        self,
        pipe: BasePipeline,
        items: List[Dict[str, Any]],
        logic: ReasoningProcessor,
        *,
        batch_size: int = 16,
        max_new_tokens: int = 256,
        K: int = 8,
    ) -> List[str]:
        """
        Self-consistency (SC) for ARC-C reasoning:
          - Generate K CoT samples with sampling (T=0.7, top_p=0.9)
          - Parse letters
          - Majority vote
          - Fallback: 'A' if all fail
        """
        prompts = logic.preprocess(items, pipe.tokenizer)

        all_votes: List[List[Optional[str]]] = []
        for _ in range(K):
            raw = pipe._generate(
                prompts,
                max_new_tokens=max(max_new_tokens, 256),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                batch_size=batch_size,
            )
            round_letters: List[Optional[str]] = []
            for txt in raw:
                letter = _extract_choice_letter(txt)
                round_letters.append(letter)
            all_votes.append(round_letters)

        final_answers: List[str] = []
        n_items = len(items)
        for i in range(n_items):
            letters: List[str] = []
            for k in range(K):
                letter = all_votes[k][i]
                if letter is not None:
                    letters.append(letter)
            if letters:
                cnt = Counter(letters)
                best, _ = cnt.most_common(1)[0]
                final_answers.append(best)
            else:
                final_answers.append("A")
        return final_answers

    def run(
        self,
        items: List[Dict[str, Any]],
        *,
        batch_size: int = 16,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:

        # 1) Classify route for each item
        texts = [_text_for_router(it) for it in items]
        routes = self.router.predict_routes(texts)

        # 2) Group items by predicted route
        buckets: Dict[str, List[int]] = {}
        for idx, rk in enumerate(routes):
            if rk not in _ALLOWED_ROUTES:
                raise ValueError(f"Unknown route from router: {rk}")
            buckets.setdefault(rk, []).append(idx)

        # 3) Run each bucket on the corresponding expert + processor
        outputs: List[str] = [""] * len(items)
        for rk, idxs in buckets.items():
            sub_items = [items[i] for i in idxs]
            pipe = self._get_expert(rk)
            logic = self.logic_map[rk]

            if rk == "reasoning":
                preds = self._run_reasoning_with_sc(
                    pipe,
                    sub_items,
                    logic=logic,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    K=self.reasoning_sc_K,
                )
            else:
                preds = pipe.run(
                    sub_items,
                    logic=logic,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )

            for i, p in zip(idxs, preds):
                outputs[i] = p

            if self.evict_after_route:
                self._evict_expert(rk)

        return outputs
