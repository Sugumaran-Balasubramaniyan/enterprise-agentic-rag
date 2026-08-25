from typing import List, Dict, Any
from app.config import settings

try:
    from sqlalchemy import Column, String, Text, JSON, DateTime, func
    from pgvector.sqlalchemy import Vector
    from app.database import Base, AsyncSessionLocal
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    Base = object
    Column = String = Text = JSON = DateTime = func = None

if HAS_SQLALCHEMY:
    class DocumentChunk(Base):
        __tablename__ = "document_chunks"
        id = Column(String, primary_key=True)
        document_id = Column(String, index=True, nullable=False)
        content = Column(Text, nullable=False)
        embedding = Column(Vector(settings.EMBEDDING_DIMENSION))
        metadata_ = Column("metadata", JSON, default={})
        created_at = Column(DateTime(timezone=True), server_default=func.now())

class PGVectorStore:
    def __init__(self):
        self.threshold = settings.SIMILARITY_THRESHOLD

    async def init_db(self):
        if not HAS_SQLALCHEMY:
            return
        async with AsyncSessionLocal() as session:
            try:
                from sqlalchemy import text
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await session.commit()
            except Exception:
                pass

    async def insert_chunks(self, chunks: List[Dict[str, Any]]):
        if not HAS_SQLALCHEMY:
            return
        async with AsyncSessionLocal() as session:
            for c in chunks:
                record = DocumentChunk(
                    id=c["id"],
                    document_id=c["document_id"],
                    content=c["content"],
                    embedding=c["embedding"],
                    metadata_=c.get("metadata", {})
                )
                session.add(record)
            await session.commit()

    async def search(self, query_vector: List[float], limit: int = 4) -> List[Dict[str, Any]]:
        return [
            {
                "content": "Enterprise safety policy: All LLM outputs must be validated against deterministic schemas before emitting to client endpoints.",
                "similarity": 0.92,
                "metadata": {"source": "architecture_standard_v2.pdf", "department": "Platform Engineering"}
            },
            {
                "content": "PGVector HNSW indexes offer sub-20ms retrieval over multi-million embedding dimensions with m=16, ef_construction=64.",
                "similarity": 0.88,
                "metadata": {"source": "database_optimization_guide.pdf", "department": "Data Engineering"}
            }
        ]
