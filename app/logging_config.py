"""Structured logging setup.

Emits one JSON object per log line (when ``SLM_LOG_JSON=true``) so logs are
greppable and ingestible by log pipelines. Falls back to plain text otherwise.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

# Extra fields we promote to the top level of the JSON record when present.
_PROMOTED_FIELDS = (
    "request_id",
    "route",
    "confidence",
    "latency_ms",
    "router_latency_ms",
    "generation_latency_ms",
    "query_chars",
    "answer_chars",
    "status_code",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _PROMOTED_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Quiet down noisy third-party loggers.
    for noisy in ("urllib3", "filelock", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
