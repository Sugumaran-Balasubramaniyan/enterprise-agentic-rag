import uuid
from fastapi import APIRouter, HTTPException
from app.api.schemas import (
    QueryRequest, QueryResponse,
    DocumentIngestRequest, DocumentIngestResponse,
    HealthResponse
)
from app.rag.chunker import RecursiveSemanticChunker
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import PGVectorStore
from app.agent.orchestrator import AgentOrchestrator
from app.config import settings

router = APIRouter()
orchestrator = AgentOrchestrator()
chunker = RecursiveSemanticChunker()
embedding_service = EmbeddingService()
vector_store = PGVectorStore()

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
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(req: DocumentIngestRequest):
    doc_id = str(uuid.uuid4())
    raw_chunks = chunker.chunk_text(req.content, metadata={**req.metadata, "title": req.title})
    
    texts = [c["content"] for c in raw_chunks]
    embeddings = await embedding_service.get_embeddings_batch(texts)
    
    chunks_to_insert = []
    for i, chunk in enumerate(raw_chunks):
        chunks_to_insert.append({
            "id": f"{doc_id}_{i}",
            "document_id": doc_id,
            "content": chunk["content"],
            "embedding": embeddings[i],
            "metadata": chunk["metadata"]
        })
        
    await vector_store.insert_chunks(chunks_to_insert)
    
    return DocumentIngestResponse(
        document_id=doc_id,
        chunks_created=len(chunks_to_insert),
        status="success"
    )
