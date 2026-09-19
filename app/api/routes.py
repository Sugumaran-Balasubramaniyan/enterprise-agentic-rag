import uuid
import math
import json
import time
import hmac
import collections
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    QueryRequest, QueryResponse,
    DocumentIngestRequest, DocumentIngestResponse,
    DocumentSummaryResponse, DocumentDetailResponse, DeleteDocumentResponse,
    SystemMetricsResponse, HealthResponse
)
from app.rag.chunker import RecursiveSemanticChunker
from app.rag.parser import EnterpriseDocumentParser
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import PGVectorStore
from app.agent.orchestrator import AgentOrchestrator
from app.config import settings
from app.observability.tracing import RequestTracer
from app.audit import get_audit_logger
from app.cache.semantic_cache import SemanticCache


class MetricsTracker:
    """In-memory telemetry tracker for query latency and safety metrics."""
    def __init__(self):
        self.total_queries: int = 0
        self.blocked_queries: int = 0
        self.latencies: List[float] = []

    def record_query(self, latency_ms: float, blocked: bool = False):
        self.total_queries += 1
        if blocked:
            self.blocked_queries += 1
        self.latencies.append(latency_ms)

    def get_stats(self) -> Dict[str, float]:
        if not self.latencies:
            return {"avg_latency_ms": 0.0, "p95_latency_ms": 0.0}

        avg_lat = sum(self.latencies) / len(self.latencies)
        sorted_lats = sorted(self.latencies)
        p95_idx = int(math.ceil(0.95 * len(sorted_lats))) - 1
        p95_idx = max(0, min(p95_idx, len(sorted_lats) - 1))
        p95_lat = sorted_lats[p95_idx]

        return {
            "avg_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(p95_lat, 2)
        }


class RateLimiter:
    """
    Small in-process sliding-window rate limiter keyed by client identifier.

    ``check`` returns ``(allowed, retry_after_seconds)``. When disabled it is a
    pure pass-through (no side effects). Not distributed / not for multi-worker
    production rate limiting; provides a safe, testable default.
    """

    def __init__(
        self,
        enabled: bool = False,
        max_requests: int = 60,
        period: int = 60,
    ) -> None:
        self.enabled = enabled
        self.max_requests = max_requests
        self.period = period
        self._clients: Dict[str, collections.deque] = {}

    def check(self, client_key: str) -> tuple:
        """
        Returns ``(allowed, retry_after_seconds)``. ``allowed`` is True when the
        request may proceed; on breach ``retry_after_seconds`` is the seconds
        until the window slides.
        """
        if not self.enabled:
            return True, 0.0
        now = time.monotonic()
        dq = self._clients.setdefault(client_key, collections.deque())
        while dq and now - dq[0] > self.period:
            dq.popleft()
        if len(dq) >= self.max_requests:
            seconds_left = max(1, int(self.period - (now - dq[0])) + 1)
            return False, float(seconds_left)
        dq.append(now)
        return True, 0.0


router = APIRouter()
vector_store = PGVectorStore()
orchestrator = AgentOrchestrator(vector_store=vector_store)
chunker = RecursiveSemanticChunker()
embedding_service = EmbeddingService()
document_parser = EnterpriseDocumentParser(default_chunker=chunker)
metrics_tracker = MetricsTracker()

# Optional enterprise hardening components (all gated behind settings flags).
semantic_cache = SemanticCache()
rate_limiter = RateLimiter(
    enabled=bool(getattr(settings, "RATE_LIMIT_ENABLED", False)),
    max_requests=int(getattr(settings, "RATE_LIMIT_REQUESTS", 60)),
    period=int(getattr(settings, "RATE_LIMIT_PERIOD_SECONDS", 60)),
)
audit_logger = get_audit_logger()


# --- Internal helpers (no-op / pass-through when flags are OFF) ------------

def _correlation_id_for(request: Request) -> str:
    header = request.headers.get("X-Correlation-ID", "")
    return (header or uuid.uuid4().hex[:12]).strip() or uuid.uuid4().hex[:12]


def require_auth_checked(request: Request) -> None:
    """
    Enforce optional bearer-token auth. Reads ``request.headers`` so TestClient
    middleware keeps working. Returns immediately (no auth) when the flag is off.
    """
    if not bool(getattr(settings, "API_AUTH_ENABLED", False)):
        return
    expected = getattr(settings, "API_AUTH_TOKEN", "")
    auth_header = request.headers.get("Authorization", "") or ""
    provided = ""
    if auth_header.startswith("Bearer "):
        provided = auth_header[len("Bearer "):].strip()
    if not expected or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


def check_rate_limit(request: Request) -> None:
    """Apply the sliding-window rate limiter when enabled (no-op otherwise)."""
    if not rate_limiter.enabled:
        return
    client_key = request.client.host if request.client else "unknown"
    allowed, retry_after = rate_limiter.check(client_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(retry_after))},
        )


def _retrieved_chunk_ids(sources: List[Any]) -> List[str]:
    ids = []
    for src in sources or []:
        if isinstance(src, dict) and src.get("id"):
            ids.append(str(src["id"]))
    return ids


def sanitize_content(content: str) -> str:
    """Optionally scrub PII at ingestion (only when the flag is enabled)."""
    if getattr(settings, "SANITIZE_PII_AT_INGEST", False) and content:
        from app.guardrails.post_execution import sanitize_pii
        return sanitize_pii(content)
    return content


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        db_connected=True
    )


@router.post("/query", response_model=QueryResponse)
async def execute_agent_query(req: QueryRequest, request: Request):
    correlation_id = _correlation_id_for(request)
    check_rate_limit(request)
    require_auth_checked(request)

    with RequestTracer(name="query", correlation_id=correlation_id) as tracer:
        # Optional semantic cache lookup.
        if bool(getattr(settings, "ENABLE_SEMANTIC_CACHE", False)):
            cached = await semantic_cache.get(req.query)
            if cached is not None:
                response = QueryResponse(**cached)
                gm = dict(response.guardrail_metrics or {})
                gm["cache_hit"] = True
                response.guardrail_metrics = gm
                tracer.record_stage("semantic_cache", hit=True)
                is_blocked = False
                metrics_tracker.record_query(response.latency_ms, blocked=False)
                audit_logger.record(
                    correlation_id=correlation_id,
                    action="query",
                    user_role=req.user_role,
                    query=req.query,
                    tool_count=len(response.tool_traces or []),
                    retrieved_chunk_ids=_retrieved_chunk_ids(response.sources),
                    guardrail_metrics=response.guardrail_metrics,
                    latency_ms=response.latency_ms,
                    blocked=False,
                )
                return response
            tracer.record_stage("semantic_cache", hit=False)

        try:
            tracer.record_stage("pre_guardrail_retrieval_generation")
            response = await orchestrator.execute(req.query, user_role=req.user_role)
            if bool(getattr(settings, "ENABLE_SEMANTIC_CACHE", False)):
                await semantic_cache.put(req.query, response.model_dump())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        is_blocked = response.guardrail_metrics.get("blocked", False) if isinstance(response.guardrail_metrics, dict) else False
        metrics_tracker.record_query(response.latency_ms, blocked=is_blocked)
        audit_logger.record(
            correlation_id=correlation_id,
            action="query",
            user_role=req.user_role,
            query=req.query,
            tool_count=len(response.tool_traces or []),
            retrieved_chunk_ids=_retrieved_chunk_ids(response.sources),
            guardrail_metrics=response.guardrail_metrics,
            latency_ms=response.latency_ms,
            blocked=is_blocked,
        )
        return response


@router.post("/query/stream")
async def execute_agent_query_stream(req: QueryRequest, request: Request):
    correlation_id = _correlation_id_for(request)
    check_rate_limit(request)
    require_auth_checked(request)

    async def sse_event_stream():
        # Semantic caching is intentionally skipped for streaming: SSE responses
        # are incremental and non-idempotent, so a single cached answer would
        # lose the event sequence clients depend on.
        tracer = RequestTracer(name="query_stream", correlation_id=correlation_id)
        tracer.__enter__()
        final_data: Dict[str, Any] = {}
        try:
            async for event_dict in orchestrator.execute_stream(req.query, user_role=req.user_role):
                event_name = event_dict.get("event", "message")
                event_data = event_dict.get("data", {})
                if event_name == "done":
                    lat = event_data.get("latency_ms", 0.0)
                    metrics_tracker.record_query(lat, blocked=False)
                    final_data = event_data
                elif event_name == "blocked":
                    lat = event_data.get("latency_ms", 0.0)
                    metrics_tracker.record_query(lat, blocked=True)
                    final_data = event_data

                yield f"event: {event_name}\ndata: {json.dumps(event_data)}\n\n"
        finally:
            tracer.__exit__(None, None, None)

        audit_logger.record(
            correlation_id=correlation_id,
            action="query_stream",
            user_role=req.user_role,
            query=req.query,
            tool_count=len(final_data.get("tool_traces", []) or []),
            retrieved_chunk_ids=_retrieved_chunk_ids(final_data.get("sources")),
            guardrail_metrics=final_data.get("guardrail_metrics", {}),
            latency_ms=final_data.get("latency_ms", 0.0),
            blocked=bool(final_data.get("guardrail_metrics", {}).get("blocked", False)),
        )

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/documents", response_model=List[DocumentSummaryResponse])
async def list_documents():
    try:
        raw_docs = await vector_store.get_all_documents()
        summaries = []
        for doc in raw_docs:
            summaries.append(DocumentSummaryResponse(
                document_id=doc.get("document_id", ""),
                title=doc.get("title", doc.get("document_id", "")),
                department=doc.get("department", ""),
                chunk_count=doc.get("chunk_count", 0),
                metadata=doc.get("metadata", {})
            ))
        return summaries
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}", response_model=DocumentDetailResponse)
async def get_document(doc_id: str):
    try:
        chunks = await vector_store.get_document_chunks(doc_id)
        if not chunks:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

        first_chunk_meta = chunks[0].get("metadata", {}) if chunks else {}
        title = first_chunk_meta.get("title") or first_chunk_meta.get("source") or doc_id
        department = first_chunk_meta.get("department", "")

        return DocumentDetailResponse(
            document_id=doc_id,
            title=title,
            department=department,
            total_chunks=len(chunks),
            chunks=chunks,
            metadata=first_chunk_meta
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_document(doc_id: str):
    try:
        deleted_count = await vector_store.delete_document(doc_id)
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

        return DeleteDocumentResponse(
            document_id=doc_id,
            chunks_deleted=deleted_count,
            status="success",
            message=f"Document '{doc_id}' deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(req: DocumentIngestRequest, request: Request):
    correlation_id = _correlation_id_for(request)
    # PII scrubbed before it reaches embeddings/vector store when enabled.
    content = sanitize_content(req.content)
    with RequestTracer(name="ingest", correlation_id=correlation_id) as tracer:
        try:
            doc_id = str(uuid.uuid4())
            raw_chunks = chunker.chunk_text(content, metadata={**req.metadata, "title": req.title})

            texts = [c["content"] for c in raw_chunks]
            embeddings = await embedding_service.get_embeddings_batch(texts) if texts else []

            chunks_to_insert = []
            for i, chunk in enumerate(raw_chunks):
                chunks_to_insert.append({
                    "id": f"{doc_id}_{i}",
                    "document_id": doc_id,
                    "content": chunk["content"],
                    "embedding": embeddings[i] if i < len(embeddings) else None,
                    "metadata": chunk["metadata"]
                })

            await vector_store.insert_chunks(chunks_to_insert)
            tracer.record_stage("chunk_embed_insert", chunks=len(chunks_to_insert))

            audit_logger.record(
                correlation_id=correlation_id,
                action="ingest",
                user_role="system",
                tool_count=0,
                retrieved_chunk_ids=[],
                guardrail_metrics={"chunks_created": len(chunks_to_insert)},
                latency_ms=0.0,
                blocked=False,
            )

            return DocumentIngestResponse(
                document_id=doc_id,
                chunks_created=len(chunks_to_insert),
                status="success"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload", response_model=DocumentIngestResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    department: Optional[str] = Form(None),
    title: Optional[str] = Form(None)
):
    correlation_id = _correlation_id_for(request)
    with RequestTracer(name="upload", correlation_id=correlation_id) as tracer:
        try:
            content_bytes = await file.read()
            content_str = content_bytes.decode("utf-8", errors="replace")
            # PII scrubbed before it reaches embeddings/vector store when enabled.
            content_str = sanitize_content(content_str)

            doc_title = title or (file.filename if file.filename else "Uploaded Document")
            meta: Dict[str, Any] = {
                "source": file.filename or "uploaded_file",
                "filename": file.filename or "uploaded_file",
                "title": doc_title,
            }
            if department:
                meta["department"] = department

            raw_chunks = document_parser.parse_and_chunk(
                content=content_str,
                filename=file.filename,
                metadata=meta
            )

            if not raw_chunks:
                raw_chunks = [{"content": content_str, "metadata": meta}]

            texts = [c["content"] for c in raw_chunks]
            embeddings = await embedding_service.get_embeddings_batch(texts) if texts else []

            doc_id = str(uuid.uuid4())
            chunks_to_insert = []
            for i, chunk in enumerate(raw_chunks):
                chunks_to_insert.append({
                    "id": f"{doc_id}_{i}",
                    "document_id": doc_id,
                    "content": chunk["content"],
                    "embedding": embeddings[i] if i < len(embeddings) else None,
                    "metadata": chunk.get("metadata", {})
                })

            await vector_store.insert_chunks(chunks_to_insert)
            tracer.record_stage("chunk_embed_insert", chunks=len(chunks_to_insert))

            audit_logger.record(
                correlation_id=correlation_id,
                action="upload",
                user_role="system",
                tool_count=0,
                retrieved_chunk_ids=[],
                guardrail_metrics={"chunks_created": len(chunks_to_insert)},
                latency_ms=0.0,
                blocked=False,
            )

            return DocumentIngestResponse(
                document_id=doc_id,
                chunks_created=len(chunks_to_insert),
                status="success"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process document upload: {str(e)}")


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    try:
        docs = await vector_store.get_all_documents()
        total_docs = len(docs)
        total_chunks = await vector_store.get_total_chunk_count()
        stats = metrics_tracker.get_stats()
        backend = "postgres" if vector_store.is_postgres_active() else "in_memory"

        return SystemMetricsResponse(
            total_queries=metrics_tracker.total_queries,
            blocked_queries=metrics_tracker.blocked_queries,
            avg_latency_ms=stats["avg_latency_ms"],
            p95_latency_ms=stats["p95_latency_ms"],
            total_documents=total_docs,
            total_chunks=total_chunks,
            active_backend=backend
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))