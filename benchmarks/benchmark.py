#!/usr/bin/env python
"""Latency / throughput benchmark for the router-expert serving API.

Sends a sample of queries to a running instance of the FastAPI service and
reports p50/p95/p99 latency and throughput, both overall and per route.

Usage
-----
    # 1. start the server (in another shell)
    uvicorn app.main:app --port 8000

    # 2. run the benchmark
    python benchmarks/benchmark.py --url http://localhost:8000 --repeats 2

Results are printed as a Markdown table and written to
``benchmarks/results/latest.json``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
DEFAULT_QUERIES = HERE / "queries.jsonl"
RESULTS_DIR = HERE / "results"


def load_queries(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(latencies: list[float]) -> dict[str, float]:
    return {
        "count": len(latencies),
        "mean_ms": round(statistics.fmean(latencies), 1),
        "p50_ms": round(percentile(latencies, 0.50), 1),
        "p95_ms": round(percentile(latencies, 0.95), 1),
        "p99_ms": round(percentile(latencies, 0.99), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
    }


def one_request(client: httpx.Client, url: str, query: str, max_new_tokens: int | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query}
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    t0 = time.perf_counter()
    resp = client.post(f"{url}/generate", json=payload, timeout=300.0)
    client_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    body = resp.json()
    return {
        "client_latency_ms": client_ms,
        "server_latency_ms": body["latency_ms"],
        "route": body["route"],
        "router_confidence": body["router_confidence"],
    }


def wait_for_health(client: httpx.Client, url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = client.get(f"{url}/health", timeout=5.0)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"Service at {url} not healthy within {timeout_s}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000", help="Base URL of the service.")
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--repeats", type=int, default=2, help="How many times to loop the query set.")
    ap.add_argument("--warmup", type=int, default=2, help="Warmup requests excluded from stats.")
    ap.add_argument("--concurrency", type=int, default=1, help="Concurrent in-flight requests.")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="Override per request.")
    ap.add_argument("--output", type=Path, default=RESULTS_DIR / "latest.json")
    args = ap.parse_args()

    queries = load_queries(args.queries)
    workload = [q for _ in range(args.repeats) for q in queries]

    with httpx.Client() as client:
        print(f"Waiting for {args.url} to become healthy...")
        wait_for_health(client, args.url, timeout_s=600.0)

        # Warmup: hit one query per route so lazy expert-load and MPS/CUDA
        # kernel-compile costs are paid before timing (excluded from stats).
        warmup_qs: list[dict[str, Any]] = []
        seen_routes: set[str] = set()
        for q in queries:
            route = q.get("expected_route")
            if route not in seen_routes:
                seen_routes.add(route)
                warmup_qs.append(q)
        warmup_qs.extend(queries[: args.warmup])
        print(f"Warming up with {len(warmup_qs)} requests...")
        for q in warmup_qs:
            one_request(client, args.url, q["query"], args.max_new_tokens)

        print(f"Running {len(workload)} requests at concurrency={args.concurrency}...")
        results: list[dict[str, Any]] = []
        wall_t0 = time.perf_counter()
        if args.concurrency <= 1:
            for q in workload:
                results.append(one_request(client, args.url, q["query"], args.max_new_tokens))
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futs = [
                    pool.submit(one_request, client, args.url, q["query"], args.max_new_tokens)
                    for q in workload
                ]
                results = [f.result() for f in futs]
        wall_s = time.perf_counter() - wall_t0

    server_lat = [r["server_latency_ms"] for r in results]
    overall = summarize(server_lat)
    throughput = round(len(results) / wall_s, 3)

    per_route: dict[str, dict[str, float]] = {}
    for route in sorted({r["route"] for r in results}):
        lat = [r["server_latency_ms"] for r in results if r["route"] == route]
        per_route[route] = summarize(lat)

    report = {
        "url": args.url,
        "concurrency": args.concurrency,
        "total_requests": len(results),
        "wall_seconds": round(wall_s, 2),
        "throughput_rps": throughput,
        "overall": overall,
        "per_route": per_route,
        "max_new_tokens_override": args.max_new_tokens,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print("\n=== Benchmark results (server-side latency) ===")
    print(f"requests={len(results)}  concurrency={args.concurrency}  "
          f"throughput={throughput} req/s  wall={report['wall_seconds']}s\n")
    header = "| scope | n | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |"
    print(header)
    print("|---|---|---|---|---|---|")
    o = overall
    print(f"| overall | {o['count']} | {o['p50_ms']} | {o['p95_ms']} | {o['p99_ms']} | {o['mean_ms']} |")
    for route, s in per_route.items():
        print(f"| {route} | {s['count']} | {s['p50_ms']} | {s['p95_ms']} | {s['p99_ms']} | {s['mean_ms']} |")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
