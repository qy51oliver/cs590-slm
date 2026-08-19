"""Pydantic request/response models for the serving API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Route = Literal["factual_qa", "reasoning", "instruction_following"]


class GenerateRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User query. The router decides which expert handles it.",
        examples=["What is the capital of France?"],
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=2048,
        description="Override the server default max_new_tokens for this request.",
    )
    temperature: Optional[float] = Field(default=None, gt=0.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)


class GenerateResponse(BaseModel):
    answer: str = Field(..., description="Expert model output.")
    route: Route = Field(..., description="Expert selected by the router.")
    router_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Softmax confidence of the routing decision."
    )
    latency_ms: float = Field(..., description="End-to-end server latency in milliseconds.")
    router_latency_ms: float = Field(..., description="Router classification latency.")
    generation_latency_ms: float = Field(..., description="Expert generation latency.")
    request_id: str = Field(..., description="Correlation id, echoed in server logs.")


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    router_loaded: bool
    experts_loaded: list[Route]
    device: str
    version: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
