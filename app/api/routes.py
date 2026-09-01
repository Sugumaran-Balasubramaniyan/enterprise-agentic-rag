import uuid
import math
import json
import time
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
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


router = APIRouter()
vector_store = PGVectorStore()
orchestrator = AgentOrchestrator(vector_store=vector_store)
chunker = RecursiveSemanticChunker()
embedding_service = EmbeddingService()
document_parser = EnterpriseDocumentParser(default_chunker=chunker)
metrics_tracker = MetricsTracker()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        db_connected=True
    )


@router.post("/query", response_model=QueryResponse)
async def execute_agent_query(req: QueryRequest):
    try:
        response = await orchestrator.execute(req.query, user_role=req.user_role)
        is_blocked = response.guardrail_metrics.get("blocked", False) if isinstance(response.guardrail_metrics, dict) else False
        metrics_tracker.record_query(response.latency_ms, blocked=is_blocked)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def execute_agent_query_stream(req: QueryRequest):
    async def sse_event_stream():
        async for event_dict in orchestrator.execute_stream(req.query, user_role=req.user_role):
            event_name = event_dict.get("event", "message")
            event_data = event_dict.get("data", {})
            if event_name == "done":
                lat = event_data.get("latency_ms", 0.0)
                metrics_tracker.record_query(lat, blocked=False)
            elif event_name == "blocked":
                lat = event_data.get("latency_ms", 0.0)
                metrics_tracker.record_query(lat, blocked=True)

            yield f"event: {event_name}\ndata: {json.dumps(event_data)}\n\n"

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
async def ingest_document(req: DocumentIngestRequest):
    doc_id = str(uuid.uuid4())
    raw_chunks = chunker.chunk_text(req.content, metadata={**req.metadata, "title": req.title})
    
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
    
    return DocumentIngestResponse(
        document_id=doc_id,
        chunks_created=len(chunks_to_insert),
        status="success"
    )


@router.post("/documents/upload", response_model=DocumentIngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    department: Optional[str] = Form(None),
    title: Optional[str] = Form(None)
):
    try:
        content_bytes = await file.read()
        content_str = content_bytes.decode("utf-8", errors="replace")

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

