"""FastAPI application exposing the router-expert SLM for inference.

Endpoints
---------
POST /generate : route a query to the right expert and return the answer.
GET  /health   : liveness/readiness probe with model + device status.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.config import get_settings
from app.inference import GenerationTimeout, ModelLoadError, RouterExpertService
from app.logging_config import configure_logging, get_logger
from app.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)

logger = get_logger("app.main")

# Populated during the lifespan startup handler; None until models are loaded.
_service: RouterExpertService | None = None
_startup_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup (never per request)."""
    global _service, _startup_error
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    logger.info("Loading router-expert models", extra={"route": None})
    t0 = time.perf_counter()
    try:
        _service = RouterExpertService(settings)
        logger.info(
            "Models ready",
            extra={"latency_ms": round((time.perf_counter() - t0) * 1000.0, 1)},
        )
    except ModelLoadError as exc:
        _startup_error = str(exc)
        logger.exception("Model load failed")
    yield
    if _service is not None:
        _service.shutdown()


app = FastAPI(
    title="Router-Expert SLM Inference API",
    description="Serving layer for a Gemma-3-270M router-expert system.",
    version=__version__,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    request_id = _request_id(request)
    logger.warning(
        "Malformed request", extra={"request_id": request_id, "status_code": 422}
    )
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="invalid_request",
            detail=str(exc.errors()),
            request_id=request_id,
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _service is None:
        return HealthResponse(
            status="error" if _startup_error else "loading",
            router_loaded=False,
            experts_loaded=[],
            device="unknown",
            version=__version__,
        )
    return HealthResponse(
        status="ok",
        router_loaded=_service.router_loaded,
        experts_loaded=_service.loaded_experts,  # type: ignore[arg-type]
        device=_service.device,
        version=__version__,
    )


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def generate(req: GenerateRequest, request: Request):
    request_id = _request_id(request)
    settings = get_settings()

    if _service is None:
        return _error(503, "model_unavailable", _startup_error, request_id)

    query = req.query.strip()
    if not query:
        return _error(422, "empty_query", "query is empty after trimming", request_id)
    if len(query) > settings.max_query_chars:
        return _error(
            422,
            "query_too_long",
            f"query exceeds {settings.max_query_chars} chars",
            request_id,
        )

    t0 = time.perf_counter()
    try:
        result = _service.generate(
            query,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    except GenerationTimeout as exc:
        logger.warning(
            "Generation timeout",
            extra={"request_id": request_id, "status_code": 504},
        )
        return _error(504, "generation_timeout", str(exc), request_id)
    except Exception as exc:  # noqa: BLE001 - surface as 500 with correlation id
        logger.exception("Generation failed", extra={"request_id": request_id})
        return _error(500, "generation_error", str(exc), request_id)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "generate",
        extra={
            "request_id": request_id,
            "route": result.route,
            "confidence": round(result.router_confidence, 4),
            "latency_ms": round(latency_ms, 1),
            "router_latency_ms": round(result.router_latency_ms, 1),
            "generation_latency_ms": round(result.generation_latency_ms, 1),
            "query_chars": len(query),
            "answer_chars": len(result.answer),
            "status_code": 200,
        },
    )
    return GenerateResponse(
        answer=result.answer,
        route=result.route,  # type: ignore[arg-type]
        router_confidence=result.router_confidence,
        latency_ms=round(latency_ms, 1),
        router_latency_ms=round(result.router_latency_ms, 1),
        generation_latency_ms=round(result.generation_latency_ms, 1),
        request_id=request_id,
    )


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or uuid.uuid4().hex[:12]


def _error(status: int, error: str, detail: str | None, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(
            error=error, detail=detail, request_id=request_id
        ).model_dump(),
    )
