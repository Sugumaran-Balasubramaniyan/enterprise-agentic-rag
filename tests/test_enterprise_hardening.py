"""
Enterprise hardening tests: RequestTracer, AuditLogger, SemanticCache,
optional API auth, and the sliding-window rate limiter. Hermetic — all paths
are tmpdir-backed and no external network calls are made.
"""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings as app_settings
from app.audit import AuditLogger
from app.cache.semantic_cache import SemanticCache
from app.observability.tracing import RequestTracer, new_correlation_id
from app.api.routes import RateLimiter


# ---------------------------------------------------------------------------
# (a) SemanticCache round-trip (deterministic mock embeddings)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_semantic_cache_hit_and_miss():
    cache = SemanticCache()  # mock embedding provider, deterministic

    await cache.put("What is HNSW?", {"answer": "a graph index", "latency_ms": 5.0})
    assert cache.misses == 0
    assert cache.hits == 0

    # Same query -> identical embedding -> cosine 1.0 -> hit.
    hit = await cache.get("What is HNSW?")
    assert hit is not None
    assert hit["answer"] == "a graph index"
    assert cache.hits == 1

    # Different query -> below threshold -> miss.
    miss = await cache.get("Tell me about retirement planning")
    assert miss is None
    assert cache.misses == 1


@pytest.mark.anyio
async def test_semantic_cache_never_raises_on_empty():
    cache = SemanticCache()
    cache.embedding_service = None  # force an internal error path defensively
    try:
        await cache.put("x", {"response": "y"})
    except Exception:
        pytest.fail("SemanticCache.put must never raise")
    result = await cache.get("")
    assert result is None


# ---------------------------------------------------------------------------
# (b) AuditLogger writes valid, non-leaking JSON lines
# ---------------------------------------------------------------------------

def test_audit_logger_writes_json_line_without_raw_query(tmp_path):
    log_path = tmp_path / "audit.log"
    raw_query = "SELECT * FROM users WHERE email = 'bob@example.com' AND secret = 'hunter2'"
    correlation_id = new_correlation_id()

    logger = AuditLogger(audit_enabled=True, path=str(log_path))
    logger.record(
        correlation_id=correlation_id,
        action="query",
        user_role="standard_user",
        query=raw_query,
        tool_count=3,
        retrieved_chunk_ids=["c1", "c2"],
        guardrail_metrics={"pre_execution_passed": True},
        latency_ms=12.5,
        blocked=False,
    )

    raw = log_path.read_text(encoding="utf-8").strip()
    record = json.loads(raw)

    # Stable audit shape + correlation id.
    assert record["correlation_id"] == correlation_id
    assert record["action"] == "query"
    assert record["user_role"] == "standard_user"

    # Query is hashed, not stored verbatim.
    assert record["query_sha256"] == hashlib.sha256(
        raw_query[:2000].encode("utf-8")
    ).hexdigest()
    assert "query" not in record
    assert "hunter2" not in raw
    assert raw_query not in raw

    assert record["tool_count"] == 3
    assert record["retrieved_chunk_ids"] == ["c1", "c2"]
    assert record["guardrail_metrics"] == {"pre_execution_passed": True}
    assert record["blocked"] is False


def test_audit_logger_respects_disabled_flag(tmp_path):
    log_path = tmp_path / "disabled.log"
    logger = AuditLogger(audit_enabled=False, path=str(log_path))
    logger.record(action="query", query="shhh", correlation_id="x")
    assert not log_path.exists()


# ---------------------------------------------------------------------------
# (c) RequestTracer records stages + correlation id round-trip
# ---------------------------------------------------------------------------

def test_request_tracer_correlation_roundtrip_and_stages():
    with RequestTracer(name="query", correlation_id="cid123") as tracer:
        tracer.record_stage("pre_guardrail", decision="pass")
        tracer.record_stage("retrieval", docs=4)

    assert tracer.correlation_id == "cid123"
    assert tracer.total_ms >= 0
    assert [s["stage"] for s in tracer.stages] == ["pre_guardrail", "retrieval"]
    assert tracer.stages[0]["decision"] == "pass"
    assert tracer.stages[1]["docs"] == 4


def test_request_tracer_generates_correlation_id():
    tracer = RequestTracer()
    assert len(tracer.correlation_id) == 12
    assert new_correlation_id() != tracer.correlation_id


def test_request_tracer_does_not_swallow_exceptions():
    with pytest.raises(RuntimeError):
        with RequestTracer(name="boom", correlation_id="cid"):
            raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# (d) Rate limiter sliding window
# ---------------------------------------------------------------------------

def test_rate_limiter_sliding_window():
    rl = RateLimiter(enabled=True, max_requests=2, period=60)
    assert rl.check("127.0.0.1") == (True, 0.0)
    assert rl.check("127.0.0.1") == (True, 0.0)
    allowed, retry_after = rl.check("127.0.0.1")
    assert allowed is False
    assert retry_after > 0
    # A different client is unaffected.
    assert rl.check("10.0.0.9") == (True, 0.0)


def test_rate_limiter_disabled_is_pass_through():
    rl = RateLimiter(enabled=False, max_requests=2, period=60)
    for _ in range(50):
        assert rl.check("anything") == (True, 0.0)


# ---------------------------------------------------------------------------
# (e) Optional API auth via bearer token (settings-flag gated)
# ---------------------------------------------------------------------------

def test_api_auth_enforced_when_enabled():
    old_enabled = getattr(app_settings, "API_AUTH_ENABLED", False)
    old_token = getattr(app_settings, "API_AUTH_TOKEN", "")
    try:
        app_settings.API_AUTH_ENABLED = True
        app_settings.API_AUTH_TOKEN = "s3cr3t-token"
        client = TestClient(__import__("app.main", fromlist=["app"]).app)

        # No auth header -> 401.
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is HNSW?", "user_role": "standard_user"},
        )
        assert resp.status_code == 401

        # Wrong token -> 401.
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is HNSW?"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

        # Correct token -> 200.
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is HNSW?"},
            headers={"Authorization": "Bearer s3cr3t-token"},
        )
        assert resp.status_code == 200
    finally:
        app_settings.API_AUTH_ENABLED = old_enabled
        app_settings.API_AUTH_TOKEN = old_token


def test_api_auth_disabled_by_default():
    old_enabled = getattr(app_settings, "API_AUTH_ENABLED", False)
    try:
        app_settings.API_AUTH_ENABLED = False
        client = TestClient(__import__("app.main", fromlist=["app"]).app)
        resp = client.post("/api/v1/query", json={"query": "What is HNSW?"})
        assert resp.status_code == 200
    finally:
        app_settings.API_AUTH_ENABLED = old_enabled


# ---------------------------------------------------------------------------
# Middleware: correlation id is echoed on the response header
# ---------------------------------------------------------------------------

def test_request_context_middleware_echoes_correlation_id():
    client = TestClient(__import__("app.main", fromlist=["app"]).app)
    resp = client.get(
        "/api/v1/health", headers={"X-Correlation-ID": "req-abc123"}
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Correlation-ID") == "req-abc123"
    assert "X-Response-Time-MS" in resp.headers