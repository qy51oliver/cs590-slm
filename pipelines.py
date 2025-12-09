from typing import Any, Dict, List, Optional

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, AutoModelForSequenceClassification
from tqdm import tqdm

# try:
#     from vllm import LLM, SamplingParams
#     VLLM_AVAILABLE = True
# except Exception:
#     VLLM_AVAILABLE = False


# ---- Router classifier (HF sequence classification) ----
_LABEL2ROUTE = {"FQA": "factual_qa", "REAS": "reasoning", "IF": "instruction_following"}
_ALLOWED_ROUTES = set(_LABEL2ROUTE.values())

class HFRouterClassifier:
    """
    Uses a Gemma-3-270M sequence-classification head trained on labels: FQA / REAS / IF.
    Predicts a route key: 'factual_qa' | 'reasoning' | 'instruction_following'.
    """
    def __init__(self, model_name, max_length=1024, head_ratio=0.7):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"  # classifier training used right padding
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True, device_map="auto"
        )
        self.model.eval()
        self.max_length = int(max_length)
        self.head_ratio = float(head_ratio)

    def _head_tail(self, ids):
        if len(ids) <= self.max_length:
            return ids
        head = int(self.max_length * self.head_ratio)
        tail = max(1, self.max_length - head)
        return ids[:head] + ids[-tail:]

    @torch.no_grad()
    def predict_routes(self, texts):
        # texts: list of raw user-facing prompts/questions
        encs = []
        for t in texts:
            ids = self.tok(t.strip(), add_special_tokens=True, truncation=False)["input_ids"]
            ids = self._head_tail(ids)
            encs.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        batch = self.tok.pad(encs, padding=True, return_tensors="pt").to(self.model.device)

        logits = self.model(**batch).logits  # [B, 3]
        pred_ids = logits.argmax(dim=-1).tolist()
        # model.config.id2label should be present from training; fallback maps 0,1,2 in order FQA/REAS/IF
        id2label = getattr(self.model.config, "id2label", {0: "FQA", 1: "REAS", 2: "IF"})
        labels = [id2label[int(i)] for i in pred_ids]
        routes = []
        for lab in labels:
            if lab not in _LABEL2ROUTE:
                raise ValueError(f"Router produced unknown label: {lab}")
            routes.append(_LABEL2ROUTE[lab])
        return routes
    
    
# ---------------- helpers ----------------
def apply_instruct_template(prompt, tokenizer: PreTrainedTokenizerBase) -> str:
    has_chat = tokenizer and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template
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
    text = re.sub(r'^\s*```.*$', '', text, flags=re.MULTILINE)
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
        dec = tokenizer.batch_decode(gen[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
        outs.extend([d.strip() for d in dec])
    return outs


# def _batched_generate_vllm(
#     model: Any,
#     tokenizer: PreTrainedTokenizerBase,
#     prompts: List[str],
#     *,
#     max_new_tokens: int = 128,
#     do_sample: bool = False,
#     temperature: float = 0.7,
#     top_p: float = 0.95,
# ) -> List[str]:
#     sampling_params = SamplingParams(
#         max_tokens=max_new_tokens,
#         temperature=temperature if do_sample else 0.0,
#         top_p=top_p if do_sample else 1.0,
#         skip_special_tokens=True,
#     )
#     outputs = model.generate(prompts, sampling_params)
#     return [o.outputs[0].text.strip() for o in outputs]


# ---------------- logic-only processors ----------------
class ProcessLogic:
    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        return [apply_instruct_template(str(it.get("prompt", "")), tokenizer) for it in items]

    def postprocess(self, outputs: List[str], items: List[Dict[str, Any]]) -> List[str]:
        # default: strip code fences and whitespace
        return [_strip_code_fences(t).strip() for t in outputs]


class FactualQAProcessor(ProcessLogic):
    def __init__(self, *, few_shot: int = 0):
        self.few_shot = max(0, min(few_shot, 5))

    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        prompts: List[str] = []
        for it in items:
            q = it.get("question") or it.get("query") or ""
            if self.few_shot <= 0:
                messages = [{"role": "user", "content": f"Answer the question concisely.\nQuestion: {q}\nAnswer:"}]
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
        return [c.split("\n")[0].strip() for c in cleaned]


class ReasoningProcessor(ProcessLogic):
    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        prompts: List[str] = []
        for it in items:
            question = it.get("question", "")
            content = (
                "You will be given a multiple-choice question with options A–D.\n"
                "Please reason and choose the correct answer **strictly as a single capital letter** (A, B, C, D). for the following question:\n"
                f"{question.strip()}\n\nAnswer:"
            )
            # content =  "Please reason and choose the correct answer for the following question:\n"
            # content += question.strip()
            prompts.append(apply_instruct_template([{ "role": "user", "content": content }], tokenizer))
        return prompts





class InstructionFollowingProcessor(ProcessLogic):
    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        texts = []
        for it in items:
            txt = it.get("prompt")
            if not txt:
                txt = it.get("question") or it.get("query") or ""
            texts.append(str(txt))
        return [apply_instruct_template(t, tokenizer) for t in texts]


# ---------------- base inference pipeline (loads model, uses processors) ----------------
class BasePipeline:
    def __init__(self, model_name: str):
        print(f"[BasePipeline] Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        self.tokenizer.padding_side = "left"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _generate(self, prompts: List[str], *, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float, batch_size: int) -> List[str]:
        # Prompts are already formatted; do not apply chat template here
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




class RouterPipeline(BasePipeline):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def _router(self, item: Dict[str, Any]) -> str:
        """
        if you would like to use a router-based pipeline, you need to implement this function and train your own router
        For now, we just return the task type as illustration, you can NOT use the task type to route the pipeline
        """
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

        # Default logic map if not provided
        if logic_map is None:
            logic_map = {
                "factual_qa": FactualQAProcessor(),
                "reasoning": ReasoningProcessor(),
                "instruction_following": InstructionFollowingProcessor(),
            }

        outputs: List[Optional[str]] = [None] * len(items)
        for key, idxs in groups.items():
            sub_items = [items[i] for i in idxs]
            logic = logic_map.get(key)
            if logic is None:
                logic = FactualQAProcessor()
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

    def run(self, items: List[Dict[str, Any]],
            logic: Optional[ProcessLogic] = None,
            **gen_kwargs) -> List[str]:
        return self.run_with_router(items, self._router, **gen_kwargs)



def _text_for_router(item):
    for k in ("question", "prompt", "query"):
        if item.get(k):
            return str(item[k])
    raise ValueError("Item missing text field: expected one of ['question','prompt','query'].")

class OurPipeline:
    """
    Router + 3 experts (lazy-loaded).
    - router_model: HF repo/path of the classifier (sequence classification head)
    - experts: dict {'factual_qa','reasoning','instruction_following'} -> HF repo/path for each expert
    - router_max_len: head+tail truncated context length used by the router encoder
    - evict_after_route: if True, unload each expert after processing its bucket to cap VRAM
    """
    def __init__(
        self,
        router_model: Optional[str] = None,
        experts: Optional[Dict[str, str]] = None,
        router_max_len: int = 1024,
        evict_after_route: bool = False,
    ):
        # Respect caller args; only fall back if None
        if router_model is None:
            router_model = "oliveryql/gemma270m-sft-router"

        if experts is None:
            experts = {
                "factual_qa": "oliveryql/gemma270m-sft-fqa",
                "reasoning": "oliveryql/gemma270m-sft-reasoning",
                "instruction_following": "google/gemma-3-270m-it",
            }

        missing = _ALLOWED_ROUTES - set(experts.keys())
        if missing:
            raise ValueError(f"experts missing routes: {sorted(missing)}")

        # Router (small classifier)
        self.router = HFRouterClassifier(router_model, max_length=router_max_len)

        # Store expert model names; defer loading until actually needed
        self._expert_names: Dict[str, str] = dict(experts)
        self._expert_pipes: Dict[str, Optional[BasePipeline]] = {k: None for k in experts.keys()}

        # One processor per route
        self.logic_map = {
            "factual_qa": FactualQAProcessor(),
            "reasoning": ReasoningProcessor(),
            "instruction_following": InstructionFollowingProcessor(),
        }

        self.evict_after_route = bool(evict_after_route)

    def _get_expert(self, route_key: str) -> BasePipeline:
        pipe = self._expert_pipes.get(route_key)
        if pipe is not None:
            return pipe
        model_name = self._expert_names[route_key]
        pipe = BasePipeline(model_name)  # loads tokenizer+model with device_map="auto"
        self._expert_pipes[route_key] = pipe
        return pipe

    def _evict_expert(self, route_key: str):
        pipe = self._expert_pipes.get(route_key)
        if pipe is None:
            return
        # Best-effort cleanup
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
        # 1) Route each item using the classifier
        texts = [_text_for_router(it) for it in items]
        routes = self.router.predict_routes(texts)

        # 2) Bucket indices by route
        buckets: Dict[str, List[int]] = {}
        for idx, rk in enumerate(routes):
            if rk not in _ALLOWED_ROUTES:
                raise ValueError(f"Unknown route from router: {rk}")
            buckets.setdefault(rk, []).append(idx)

        # 3) Run each bucket on its expert with the matching processor
        outputs: List[str] = [""] * len(items)
        for rk, idxs in buckets.items():
            sub_items = [items[i] for i in idxs]
            pipe = self._get_expert(rk)
            logic = self.logic_map[rk]
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