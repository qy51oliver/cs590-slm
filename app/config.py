"""Runtime configuration, sourced from environment variables.

All settings can be overridden with env vars prefixed with ``SLM_`` (e.g.
``SLM_ROUTER_MODEL``, ``SLM_MAX_NEW_TOKENS``). Defaults reproduce the
router-expert system described in the project report.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SLM_",
        env_file=".env",
        extra="ignore",
    )

    # --- Model identifiers (HF hub ids or local paths) ---
    router_model: str = Field(
        default="oliveryql/gemma270m-sft-router",
        description="Sequence-classification router checkpoint.",
    )
    fqa_model: str = Field(default="oliveryql/gemma270m-sft-fqa")
    reasoning_model: str = Field(default="oliveryql/gemma270m-sft-reasoning")
    if_model: str = Field(default="oliveryql/gemma270m-sft-if")

    # --- Routing / decoding defaults ---
    router_max_len: int = Field(default=1024, ge=1)
    reasoning_sc_k: int = Field(
        default=5, ge=1, description="Self-consistency samples for the reasoning expert."
    )
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.44, gt=0.0)
    top_p: float = Field(default=0.6, gt=0.0, le=1.0)

    # --- Serving behaviour ---
    warm_experts: bool = Field(
        default=False,
        description="Load all three experts at startup instead of lazily on first use.",
    )
    generation_timeout_s: float = Field(
        default=60.0, gt=0.0, description="Per-request wall-clock budget for generation."
    )
    max_query_chars: int = Field(default=8000, ge=1)

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True, description="Emit structured JSON logs.")

    @property
    def experts(self) -> dict[str, str]:
        return {
            "factual_qa": self.fqa_model,
            "reasoning": self.reasoning_model,
            "instruction_following": self.if_model,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
