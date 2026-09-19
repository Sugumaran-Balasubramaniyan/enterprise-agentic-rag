"""
Append-only JSON-lines audit logging.

Each ``record(...)`` call appends a single JSON object to an append-only log
file. Raw query text is hashed (SHA-256) before logging and is NEVER written
verbatim unless the ``AUDIT_LOG_QUERY=true`` environment variable is set. All
writes are serialized with a ``threading.Lock`` and flushed line-by-line.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.config import settings


def _env_truthy(key: str) -> bool:
    return str(os.getenv(key, "")).strip().lower() in {"1", "true", "yes", "on"}


class AuditLogger:
    """
    Append-only JSON-lines audit logger.

    `record` fields are normalized into a stable shape that always contains:
    ts, correlation_id, action, user_role, query_sha256, tool_count,
    retrieved_chunk_ids, guardrail_metrics, latency_ms, blocked.
    """

    def __init__(
        self,
        audit_enabled: Optional[bool] = None,
        path: Optional[str] = None,
    ) -> None:
        self.enabled = (
            audit_enabled
            if audit_enabled is not None
            else bool(getattr(settings, "AUDIT_LOG_ENABLED", True))
        )
        self.path = path or os.getenv("AUDIT_LOG_PATH", "logs/audit.log")
        self._lock = threading.Lock()
        self._fh = None
        if self.enabled:
            try:
                full = os.path.abspath(self.path)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                self._fh = open(full, "a", encoding="utf-8")
            except Exception:
                self._fh = None

    def record(self, **fields: Any) -> None:
        """Append one audit record (JSON line). Never raises."""
        if not self.enabled or self._fh is None:
            return
        line = self._serialize(fields)
        try:
            with self._lock:
                self._fh.write(line + "\n")
                self._fh.flush()
        except Exception:
            pass

    # -- Internals ----------------------------------------------------------

    def _serialize(self, fields: Dict[str, Any]) -> str:
        record: Dict[str, Any] = {"ts": datetime.now(timezone.utc).isoformat()}
        record["correlation_id"] = fields.get("correlation_id")
        record["action"] = fields.get("action")
        record["user_role"] = fields.get("user_role")

        query = fields.get("query")
        if query is not None:
            truncated = str(query)[:2000]
            record["query_sha256"] = hashlib.sha256(
                truncated.encode("utf-8")
            ).hexdigest()
            if _env_truthy("AUDIT_LOG_QUERY"):
                # Opt-in only: never log raw queries by default.
                record["query"] = truncated
        else:
            record["query_sha256"] = None

        record["tool_count"] = fields.get("tool_count", 0)
        record["retrieved_chunk_ids"] = fields.get("retrieved_chunk_ids") or []
        record["guardrail_metrics"] = fields.get("guardrail_metrics") or {}
        record["latency_ms"] = fields.get("latency_ms")
        record["blocked"] = bool(fields.get("blocked", False))
        return json.dumps(record, default=str)


_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Return the module-level singleton audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger