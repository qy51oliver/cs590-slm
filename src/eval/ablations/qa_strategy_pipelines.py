from typing import Any, Dict, List, Optional
import re
from collections import Counter

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from tqdm import tqdm


# ---------------- helpers ----------------
def apply_instruct_template(prompt: Any, tokenizer: PreTrainedTokenizerBase) -> str:
    """
    Turn a user prompt or messages list into a single string using the model's chat_template if available.
    
    Args:
        prompt: The user prompt (str) or a list of message dicts ({"role": ..., "content": ...}).
        tokenizer: The Hugging Face tokenizer object.

    Returns:
        The formatted prompt string suitable for model input.
    """
    has_chat = tokenizer and hasattr(tokenizer, "chat_template") and tokenizer.chat_template
    if has_chat:
        # Use the tokenizer's chat template if available (preferred for instruct models)
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Fallback to simple formatting if no chat template is found
    if isinstance(prompt, list):
        lines: List[str] = []
        for m in prompt:
            role = m.get("role", "user")
            prefix = "User" if role == "user" else ("Assistant" if role == "assistant" else role.capitalize())
            lines.append(f"{prefix}: {m.get('content', '')}")
        return "\n".join(lines)
    
    return str(prompt)


def _strip_code_fences(text: str) -> str:
    """
    Strips markdown code fences (```...) from the beginning/end of a generated text.

    Args:
        text: The raw generated string.

    Returns:
        The string with code fences removed.
    """
    text = re.sub(r"^\s*```.*$", "", text, flags=re.MULTILINE)
    return text.strip()


@torch.no_grad()
def _batched_generate_hf(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    *,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 16,
) -> List[str]:
    """
    Performs batched text generation using a Hugging Face model.

    Args:
        model: The loaded Hugging Face model.
        tokenizer: The Hugging Face tokenizer.
        prompts: A list of prompt strings.
        max_new_tokens: The maximum number of tokens to generate.
        do_sample: Whether to use sampling (True) or greedy decoding (False).
        temperature: Sampling temperature (only used if do_sample=True).
        top_p: Top-p value for nucleus sampling (only used if do_sample=True).
        batch_size: The batch size for generation.

    Returns:
        A list of generated strings, post-processed to remove code fences.
    """
    outs: List[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        # Tokenize the batch, ensuring padding and truncation
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # Generate the tokens
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Decode the generated part (excluding the input prompt tokens)
        dec = tokenizer.batch_decode(gen[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
        
        # Post-process and extend results
        outs.extend([_strip_code_fences(d.strip()) for d in dec])
    return outs


# ---------------- QA prompt logic ----------------
class FactualQAProcessor:
    """
    Simple TriviaQA-style prompt builder.
    'Answer the question concisely.'
    """

    def __init__(self, *, few_shot_examples: Optional[List[Dict[str, str]]] = None):
        # each example: {"question": "...", "answer": "..."}
        self.few_shot_examples = few_shot_examples or []

    def build_prompt(self, question: str, tokenizer: PreTrainedTokenizerBase) -> str:
        """
        Constructs the full prompt string for a single QA instance, including few-shot examples.

        Args:
            question: The question string.
            tokenizer: The Hugging Face tokenizer for template application.

        Returns:
            The complete, formatted prompt string.
        """
        parts: List[str] = ["Answer the question concisely."]
        
        # Add few-shot examples
        for ex in self.few_shot_examples:
            ex_q = ex.get("question", "")
            ex_a = ex.get("answer", "")
            parts.append(f"Question: {ex_q}\nAnswer: {ex_a}")
        
        # Add the current question prompt
        parts.append(f"Question: {question}\nAnswer:")
        
        # Format as a chat message (user role)
        messages = [{"role": "user", "content": "\n".join(parts)}]
        return apply_instruct_template(messages, tokenizer)

    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        """
        Prepares a list of prompt strings from a list of QA item dicts.

        Args:
            items: A list of dicts, each containing a "question" or "query".
            tokenizer: The Hugging Face tokenizer.

        Returns:
            A list of formatted prompt strings.
        """
        prompts: List[str] = []
        for it in items:
            q = it.get("question") or it.get("query") or ""
            prompts.append(self.build_prompt(str(q), tokenizer))
        return prompts

    def postprocess(self, outputs: List[str]) -> List[str]:
        """
        Post-processes raw model outputs, keeping only the first line as the concise answer.

        Args:
            outputs: A list of raw generated strings.

        Returns:
            A list of processed answer strings.
        """
        # only keep the first line as the answer
        return [o.splitlines()[0].strip() if o else "" for o in outputs]


# ---------------- base generation pipeline ----------------
class BaseFQAPipeline:
    """
    Minimal pipeline wrapping a single FQA expert model, with multiple inference strategies.
    """

    def __init__(self, model_name: str = "oliveryql/gemma270m-sft-fqa"):
        print(f"[FQAPipeline] Loading model: {model_name}")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Set tokenizer/model config for batch processing
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.logic = FactualQAProcessor()

    # ---------- shared prompt builder ----------
    def _generate_prompts(
        self,
        items: List[Dict[str, Any]],
    ) -> List[str]:
        """Generates the formatted prompts for all items."""
        return self.logic.preprocess(items, self.tokenizer)

    # ---------- Strategy 1: simple greedy ----------
    def _run_greedy(
        self,
        items: List[Dict[str, Any]],
        *,
        max_new_tokens: int = 64,
        batch_size: int = 16,
    ) -> List[str]:
        """Runs greedy decoding (no sampling)."""
        prompts = self._generate_prompts(items)
        raw = _batched_generate_hf(
            self.model,
            self.tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False, # Force greedy
            temperature=1.0, # Ignored but set
            top_p=1.0, # Ignored but set
            batch_size=batch_size,
        )
        return self.logic.postprocess(raw)

    # ---------- Strategy 2: multi-sample + vote ----------
    def _run_sample_multi(
        self,
        items: List[Dict[str, Any]],
        *,
        K: int = 3,
        max_new_tokens: int = 64,
        batch_size: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Sample K answers and take majority vote after normalization.
        
        Args:
            items: List of QA items.
            K: Number of samples to generate per item.
            max_new_tokens: Max tokens for generation.
            batch_size: Batch size for generation.
            temperature: Sampling temperature.
            top_p: Top-p value.
        
        Returns:
            A list of final answers chosen by majority vote.
        """
        prompts = self._generate_prompts(items)
        all_rounds: List[List[str]] = []
        
        # Generate K samples for all prompts
        for _ in range(K):
            raw = _batched_generate_hf(
                self.model,
                self.tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
            )
            all_rounds.append(self.logic.postprocess(raw))

        def _norm(s: str) -> str:
            """Normalizes a string for robust voting (lowercase, remove punctuation, reduce whitespace)."""
            s = s.lower()
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        final: List[str] = []
        n = len(items)
        
        # Tally votes for each item
        for i in range(n):
            votes: List[str] = []
            for r in range(K):
                ans = all_rounds[r][i]
                votes.append(_norm(ans))
            
            cnt = Counter(votes)
            best_norm, _ = cnt.most_common(1)[0]
            
            # pick the first original string that matches best_norm
            chosen = ""
            for r in range(K):
                if _norm(all_rounds[r][i]) == best_norm:
                    chosen = all_rounds[r][i]
                    break
            final.append(chosen)
            
        return final

    # ---------- Strategy 3: full two-stage refine (baseline,整体效果不好，但保留用于对比) ----------
    def _run_refine_two_stage(
        self,
        items: List[Dict[str, Any]],
        *,
        draft_max_new_tokens: int = 64,
        refine_max_new_tokens: int = 16,
        batch_size: int = 16,
    ) -> List[str]:
        """
        Two-stage QA (used as a baseline; in your results this hurts F1 if applied to all samples):
        1) Draft answer with standard FQA prompt (greedy, short).
        2) Refine answer with an explicit 'refine' prompt to make it concise and exact.
        """
        # Stage 1: draft
        draft_answers = self._run_greedy(
            items,
            max_new_tokens=draft_max_new_tokens,
            batch_size=batch_size,
        )

        # Stage 2: refine
        refine_items: List[Dict[str, Any]] = []
        for it, a1 in zip(items, draft_answers):
            q = it.get("question") or it.get("query") or ""
            refine_prompt = (
                "Refine the answer to ensure it is concise and contains only the exact factual answer.\n"
                "Do not add explanations.\n\n"
                f"Question: {q}\n"
                f"Draft answer: {a1}\n"
                "Final concise answer:"
            )
            # Construct chat-style messages directly here for apply_instruct_template
            refine_items.append({"prompt": [{"role": "user", "content": refine_prompt}]})

        prompts: List[str] = []
        for it in refine_items:
            messages = it["prompt"]
            prompts.append(apply_instruct_template(messages, self.tokenizer))

        # Run generation for the refinement prompts (greedy)
        raw = _batched_generate_hf(
            self.model,
            self.tokenizer,
            prompts,
            max_new_tokens=refine_max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            batch_size=batch_size,
        )
        
        # Simple post-process: first line only
        return [o.splitlines()[0].strip() if o else "" for o in raw]

    # ---------- Strategy 4: 组合策略——仅对“长答案”再 refine 一次 ----------
    def _run_combo_greedy_refine_if_long(
        self,
        items: List[Dict[str, Any]],
        *,
        greedy_max_new_tokens: int = 64,
        refine_max_new_tokens: int = 16,
        length_threshold: int = 8,
        batch_size: int = 16,
    ) -> List[str]:
        """
        Combination strategy:
        1) Run greedy_64 on all examples (this is your best single strategy so far).
        2) For answers that are "too long" (e.g., > length_threshold tokens),
           run a two-stage refine ONLY on those items and replace their answers.

        Intuition:
        - Avoid refining answers that are already short/acceptable (full refine degraded F1 in experiments);
        - Only attempt to compress answers that are clearly verbose, making them closer to the TriviaQA golden answer format.
        
        Args:
            items: List of QA items.
            greedy_max_new_tokens: Max tokens for initial greedy draft.
            refine_max_new_tokens: Max tokens for the refinement stage.
            length_threshold: Word count threshold to consider an answer "too long" for refinement.
            batch_size: Batch size for generation.

        Returns:
            A list of final answers, with long ones potentially refined.
        """
        # Step 1: greedy draft for all
        draft_answers = self._run_greedy(
            items,
            max_new_tokens=greedy_max_new_tokens,
            batch_size=batch_size,
        )
        final_answers = list(draft_answers)

        # Identify "too long" answers
        long_indices: List[int] = []
        for i, ans in enumerate(draft_answers):
            # Check length based on word count
            if len(ans.split()) > length_threshold:
                long_indices.append(i)

        if not long_indices:
            # No long answers, return greedy results directly
            return final_answers

        # Run two-stage refine ONLY on the subset of long answers
        sub_items = [items[i] for i in long_indices]
        refined_sub = self._run_refine_two_stage(
            sub_items,
            draft_max_new_tokens=greedy_max_new_tokens,
            refine_max_new_tokens=refine_max_new_tokens,
            batch_size=batch_size,
        )

        # Replace the original long answers with the refined ones
        for idx, ref in zip(long_indices, refined_sub):
            # Only replace if refinement produced a non-empty answer
            final_answers[idx] = ref or final_answers[idx]

        return final_answers

    # ---------- Public Entry Point ----------
    def run_strategy(
        self,
        items: List[Dict[str, Any]],
        *,
        strategy: str = "greedy_64",
        batch_size: int = 16,
    ) -> List[str]:
        """
        Runs the specified inference strategy on the provided QA items.
        
        Supported strategies:
            - greedy_64            : greedy, max_new_tokens=64 (currently the best single strategy in experiments)
            - greedy_128           : greedy, max_new_tokens=128
            - refine_32_16         : two-stage, draft 32, refine 16 (performed poorly overall in experiments)
            - refine_64_16         : two-stage, draft 64, refine 16 (performed poorly overall)
            - combo_greedy_refine_if_long: 
                * Initial greedy_64
                * Two-stage refine ONLY on answers with word count > length_threshold
            - sample3              : sample K=3, vote

        Args:
            items: List of QA items.
            strategy: The name of the inference strategy to use.
            batch_size: Batch size for generation.

        Returns:
            A list of final answer strings.
        
        Raises:
            ValueError: If an unknown strategy is specified.
        """
        strategy = strategy.lower()
        if strategy == "greedy_64":
            return self._run_greedy(items, max_new_tokens=64, batch_size=batch_size)
        if strategy == "greedy_128":
            return self._run_greedy(items, max_new_tokens=128, batch_size=batch_size)
        if strategy == "refine_32_16":
            return self._run_refine_two_stage(
                items,
                draft_max_new_tokens=32,
                refine_max_new_tokens=16,
                batch_size=batch_size,
            )
        if strategy == "refine_64_16":
            return self._run_refine_two_stage(
                items,
                draft_max_new_tokens=64,
                refine_max_new_tokens=16,
                batch_size=batch_size,
            )
        if strategy == "combo_greedy_refine_if_long":
            return self._run_combo_greedy_refine_if_long(
                items,
                greedy_max_new_tokens=64,
                refine_max_new_tokens=16,
                length_threshold=8,
                batch_size=batch_size,
            )
        if strategy == "sample3":
            return self._run_sample_multi(
                items,
                K=3,
                max_new_tokens=64,
                batch_size=batch_size,
                temperature=0.7,
                top_p=0.9,
            )

        raise ValueError(f"Unknown strategy: {strategy}")