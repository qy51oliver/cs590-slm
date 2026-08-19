"""Serving-oriented wrapper around the router-expert pipeline.

This composes the existing :class:`OurPipeline` (learned router + three task
experts + reasoning self-consistency) and exposes a single-request ``generate``
method that returns the answer together with routing metadata and latency
breakdown.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import List

from app.config import Settings
from app.logging_config import get_logger
from src.inference.pipeline import OurPipeline

logger = get_logger("app.inference")


class ModelLoadError(RuntimeError):
    """Raised when the router or an expert fails to load."""


class GenerationTimeout(RuntimeError):
    """Raised when generation exceeds the configured wall-clock budget."""


@dataclass
class GenerationResult:
    answer: str
    route: str
    router_confidence: float
    router_latency_ms: float
    generation_latency_ms: float


class RouterExpertService:
    """Thread-safe-enough single-process inference service.

    Generation is serialized through a single-worker executor so we can enforce
    a wall-clock timeout without concurrent access to the underlying models
    (Transformers models are not safe to call concurrently from many threads).
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slm-gen")
        try:
            self._pipe = OurPipeline(
                router_model=settings.router_model,
                experts=settings.experts,
                router_max_len=settings.router_max_len,
                reasoning_sc_K=settings.reasoning_sc_k,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise ModelLoadError(f"Failed to load router/experts: {exc}") from exc

        if settings.warm_experts:
            for route in settings.experts:
                logger.info("Warming expert", extra={"route": route})
                self._pipe._get_expert(route)

    # --- introspection for /health ---
    @property
    def router_loaded(self) -> bool:
        return getattr(self._pipe, "router", None) is not None

    @property
    def loaded_experts(self) -> List[str]:
        return [r for r, p in self._pipe._expert_pipes.items() if p is not None]

    @property
    def device(self) -> str:
        try:
            return str(self._pipe.router.model.device)
        except Exception:  # pragma: no cover
            return "unknown"

    # --- core inference ---
    def _generate_sync(
        self,
        query: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerationResult:
        item = {"question": query}

        t0 = time.perf_counter()
        route, confidence = self._pipe.router.predict_routes_with_scores([query])[0]
        router_ms = (time.perf_counter() - t0) * 1000.0

        expert = self._pipe._get_expert(route)
        logic = self._pipe.logic_map[route]

        t1 = time.perf_counter()
        if route == "reasoning":
            preds = self._pipe._run_reasoning_with_sc(
                expert,
                [item],
                logic=logic,
                batch_size=1,
                max_new_tokens=max_new_tokens,
                K=self._pipe.reasoning_sc_K,
            )
        else:
            preds = expert.run(
                [item],
                logic=logic,
                batch_size=1,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=temperature,
                top_p=top_p,
            )
        gen_ms = (time.perf_counter() - t1) * 1000.0

        return GenerationResult(
            answer=preds[0],
            route=route,
            router_confidence=confidence,
            router_latency_ms=router_ms,
            generation_latency_ms=gen_ms,
        )

    def generate(
        self,
        query: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> GenerationResult:
        s = self._settings
        future = self._executor.submit(
            self._generate_sync,
            query,
            max_new_tokens=max_new_tokens or s.max_new_tokens,
            temperature=temperature if temperature is not None else s.temperature,
            top_p=top_p if top_p is not None else s.top_p,
        )
        try:
            return future.result(timeout=s.generation_timeout_s)
        except FutureTimeout as exc:
            raise GenerationTimeout(
                f"Generation exceeded {s.generation_timeout_s}s budget"
            ) from exc

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
