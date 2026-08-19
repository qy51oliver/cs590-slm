# Serving layer

A FastAPI service ([`app/`](../app/)) wraps the router-expert system for online inference:
models load **once at startup**, requests are typed with Pydantic, every request is logged
as structured JSON with a latency breakdown, and failure modes return typed errors.

| File | Responsibility |
|------|----------------|
| [`app/main.py`](../app/main.py) | Endpoints, lifespan model loading, error handlers |
| [`app/inference.py`](../app/inference.py) | `RouterExpertService`: routing + dispatch + generation-timeout wrapper |
| [`app/config.py`](../app/config.py) | Env-var configuration (pydantic-settings) |
| [`app/schemas.py`](../app/schemas.py) | Pydantic request/response models |
| [`app/logging_config.py`](../app/logging_config.py) | Structured JSON logging |

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/generate` | Route a query, run the selected expert, return answer + routing metadata + latency. |
| `GET`  | `/health` | Liveness/readiness: router loaded, which experts are loaded, device, version. |
| `GET`  | `/docs` | Auto-generated Swagger UI. |

### `POST /generate`

```bash
curl -s http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"query": "List three benefits of regular exercise. Use exactly three bullet points.",
       "max_new_tokens": 128}'
```

```json
{
  "answer": "- Improved cardiovascular health by increasing blood flow to the muscles...\n- ...",
  "route": "instruction_following",
  "router_confidence": 1.0,
  "latency_ms": 1202.0,
  "router_latency_ms": 18.7,
  "generation_latency_ms": 1183.3,
  "request_id": "aff87d0bc55d"
}
```

Optional request fields: `max_new_tokens`, `temperature`, `top_p` (each overrides the
server default for that request).

### `GET /health`

```json
{
  "status": "ok",
  "router_loaded": true,
  "experts_loaded": ["factual_qa", "reasoning", "instruction_following"],
  "device": "mps:0",
  "version": "0.1.0"
}
```

### Errors

Typed `ErrorResponse` with a correlation `request_id`: `422` malformed/empty/oversized
query, `503` models not loaded, `504` generation exceeded `SLM_GENERATION_TIMEOUT_S`,
`500` unexpected generation error.

## Configuration

All via environment variables (prefix `SLM_`); see [`app/config.py`](../app/config.py):

| Variable | Default | Meaning |
|----------|---------|---------|
| `SLM_ROUTER_MODEL` / `SLM_FQA_MODEL` / `SLM_REASONING_MODEL` / `SLM_IF_MODEL` | HF hub ids | Model ids or local paths |
| `SLM_REASONING_SC_K` | 5 | Self-consistency samples for reasoning |
| `SLM_MAX_NEW_TOKENS` | 256 | Default generation length |
| `SLM_TEMPERATURE` / `SLM_TOP_P` | 0.44 / 0.6 | Decoding defaults |
| `SLM_WARM_EXPERTS` | false | Load all experts at startup vs. lazily |
| `SLM_GENERATION_TIMEOUT_S` | 60 | Per-request wall-clock budget |
| `SLM_LOG_LEVEL` / `SLM_LOG_JSON` | INFO / true | Logging |

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# first start downloads the 4 checkpoints (~2 GB) from the HF Hub
```

## Run in Docker

```bash
docker compose up --build       # builds the image and serves on :8000
# or:
docker build -t router-expert-slm .
docker run -p 8000:8000 -v hf-cache:/home/appuser/.cache/huggingface router-expert-slm
```

The [`Dockerfile`](../Dockerfile) is multi-stage (build venv → slim runtime), runs as a
non-root user, installs **CPU-only** Torch to keep the image small, and ships a
`HEALTHCHECK`. [`docker-compose.yml`](../docker-compose.yml) mounts a named volume for the
HF cache so weights download only once.

## Benchmark

Harness: [`benchmarks/benchmark.py`](../benchmarks/benchmark.py) sends a fixed set of
queries spanning all three routes and reports server-side latency percentiles + throughput,
overall and per route.

**Environment:** Apple M4 (10 cores, 24 GB), macOS 15.7.3, Python 3.12, PyTorch 2.9.1 on the
**MPS** backend, float32. All experts warmed at startup, sequential requests
(`concurrency=1`), 39 timed requests.

| Route (expert) | n | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|----------------|---|:---:|:---:|:---:|:---:|
| **overall** | 39 | 318.4 | 3319.3 | 3730.2 | 674.6 |
| factual_qa | 15 | 137.7 | 285.0 | 286.8 | 163.5 |
| reasoning *(CoT-SC, K=5)* | 12 | 321.2 | 384.7 | 427.1 | 332.3 |
| instruction_following | 12 | 1180.9 | 3645.2 | 3865.2 | 1655.8 |

Throughput: **1.48 req/s** sequential. Latency is dominated by output length: factual
answers are short; instruction-following generates up to `max_new_tokens` of free text
(widest tail); the reasoning route runs K=5 sampled CoT passes but stops early on short
letter answers. Generation is serialized behind a single-worker executor (one accelerator),
so throughput scales with sequence length rather than request concurrency. Numbers vary
±20% run-to-run on a thermally-throttled laptop; re-running regenerates
`benchmarks/results/latest.json`.

Reproduce:

```bash
uvicorn app.main:app --port 8000 &
python benchmarks/benchmark.py --url http://localhost:8000 --repeats 3
```
