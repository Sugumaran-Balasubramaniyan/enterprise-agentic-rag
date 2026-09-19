"""
Request tracing utilities.

`RequestTracer` records stage-level timing for a single request and, on exit,
emits a structured trace via structlog. The emitted attributes mirror the
OpenTelemetry GenAI semantic conventions (``gen_ai.operation.name`` and
``gen_ai.system``); a full OTel SDK is optional and NOT required here.

Optionally, when the ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_HOST`` environment
variables are both set, a minimal trace event is POSTed to Langfuse as an
asynchronous, fire-and-forget request (2s timeout, all errors swallowed). The
tracer never blocks or fails an application request on tracing.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

import structlog

from app.config import settings

logger = structlog.get_logger("rag.tracing")


def new_correlation_id() -> str:
    """Generate a short correlation/request id (uuid4 hex, 12 chars)."""
    return uuid.uuid4().hex[:12]


class RequestTracer:
    """
    Context manager that times a request and emits a structured trace on exit.

    Example:
        with RequestTracer(name="query", correlation_id="abc123") as tracer:
            tracer.record_stage("pre_guardrail", decision="pass")
            ...
    """

    def __init__(
        self,
        name: str = "request",
        correlation_id: Optional[str] = None,
    ) -> None:
        self.name = name
        self.correlation_id = correlation_id or new_correlation_id()
        self.stages: List[Dict[str, Any]] = []
        self.total_ms: float = 0.0
        self._start_time: Optional[float] = None

    def __enter__(self) -> "RequestTracer":
        self._start_time = time.perf_counter()
        return self

    def record_stage(self, stage_name: str, **attrs: Any) -> None:
        """Append a stage record with an offset timestamp (ms since enter)."""
        ts_ms = 0.0
        if self._start_time is not None:
            ts_ms = round((time.perf_counter() - self._start_time) * 1000, 3)
        record = {"stage": stage_name, "ts_ms": ts_ms}
        record.update(attrs)
        self.stages.append(record)

    def __exit__(self, exc_type, exc, tb):
        if self._start_time is not None:
            self.total_ms = round((time.perf_counter() - self._start_time) * 1000, 3)
        self._emit_trace()
        self._emit_langfuse()
        # Do not swallow exceptions raised inside the traced block.
        return False

    # -- Internals ----------------------------------------------------------

    def _emit_trace(self) -> None:
        if not bool(getattr(settings, "TRACE_ENABLED", True)):
            return
        try:
            logger.info(
                "request_trace",
                correlation_id=self.correlation_id,
                name=self.name,
                stages=self.stages,
                total_ms=self.total_ms,
                # OpenTelemetry GenAI semantic-convention attributes.
                **{"gen_ai.operation.name": self.name, "gen_ai.system": "custom"},
            )
        except Exception:
            # Tracing must never break the request.
            pass

    def _emit_langfuse(self) -> None:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST")
        if not public_key or not host:
            return
        try:
            import httpx as _httpx

            base = host.rstrip("/")
            payload = {
                "name": self.name,
                "traceId": self.correlation_id,
                "timestamp": time.time() * 1000.0,
                "input": {"correlation_id": self.correlation_id},
                "metadata": {
                    "stages": self.stages,
                    "total_ms": self.total_ms,
                    "gen_ai.operation.name": self.name,
                    "gen_ai.system": "custom",
                },
            }
            _httpx.post(
                f"{base}/api/traces",
                json=payload,
                headers={"Authorization": f"Bearer {public_key}"},
                timeout=2.0,
            )
        except Exception:
            # Fire-and-forget: never fail or block the request on tracing.
            pass