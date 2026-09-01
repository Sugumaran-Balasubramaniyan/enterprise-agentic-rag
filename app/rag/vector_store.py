import math
import re
import uuid
from typing import List, Dict, Any, Optional
from app.config import settings
from app.rag.embeddings import EmbeddingService

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sqlalchemy import Column, String, Text, JSON, DateTime, func, select, delete, text
    from pgvector.sqlalchemy import Vector
    from app.database import Base, AsyncSessionLocal, engine
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    Base = object
    Column = String = Text = JSON = DateTime = func = select = delete = text = None
    AsyncSessionLocal = None
    engine = None

if HAS_SQLALCHEMY and Base is not object:
    class DocumentChunk(Base):
        __tablename__ = "document_chunks"
        id = Column(String, primary_key=True)
        document_id = Column(String, index=True, nullable=False)
        content = Column(Text, nullable=False)
        embedding = Column(Vector(settings.EMBEDDING_DIMENSION))
        metadata_ = Column("metadata", JSON, default=dict)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
else:
    DocumentChunk = None


DEFAULT_ENTERPRISE_DOCUMENTS = [
    {
        "id": "chunk_platform_arch_001",
        "document_id": "doc_arch_standards",
        "content": (
            "Enterprise safety policy: All LLM outputs must be validated against deterministic schemas "
            "before emitting to client endpoints. Microservice architectures require mutual TLS (mTLS) "
            "and zero-trust service mesh authentication across EU regions."
        ),
        "metadata": {
            "source": "architecture_standard_v2.pdf",
            "department": "Platform Engineering",
            "title": "Platform Engineering Architecture Standards",
            "category": "Architecture & Governance",
            "version": "2.4",
            "author": "Platform Architecture Guild"
        }
    },
    {
        "id": "chunk_platform_arch_002",
        "document_id": "doc_arch_standards",
        "content": (
            "High availability deployment topology: Services must deploy across multiple availability zones "
            "in eu-west-3 with automated failover, circuit breakers, and sub-100ms P99 latency SLA targets."
        ),
        "metadata": {
            "source": "architecture_standard_v2.pdf",
            "department": "Platform Engineering",
            "title": "Platform Engineering Architecture Standards",
            "category": "Deployment & Resiliency",
            "version": "2.4",
            "author": "Platform Architecture Guild"
        }
    },
    {
        "id": "chunk_pgvector_hnsw_001",
        "document_id": "doc_pgvector_hnsw",
        "content": (
            "PGVector HNSW indexes offer sub-20ms retrieval over multi-million embedding dimensions with "
            "m=16, ef_construction=64. Vector distance metrics should use cosine distance for normalized embeddings."
        ),
        "metadata": {
            "source": "database_optimization_guide.pdf",
            "department": "Data Engineering",
            "title": "PGVector HNSW Indexing & Optimization Guide",
            "category": "Database Performance",
            "version": "1.8",
            "author": "Data Platform Team"
        }
    },
    {
        "id": "chunk_pgvector_hnsw_002",
        "document_id": "doc_pgvector_hnsw",
        "content": (
            "Hybrid search strategies in PostgreSQL: Combining pgvector cosine similarity distance with tsvector "
            "lexical keyword matching using Reciprocal Rank Fusion (RRF with k=60) provides superior recall for technical queries."
        ),
        "metadata": {
            "source": "database_optimization_guide.pdf",
            "department": "Data Engineering",
            "title": "PGVector HNSW Indexing & Optimization Guide",
            "category": "Hybrid Retrieval",
            "version": "1.8",
            "author": "Data Platform Team"
        }
    },
    {
        "id": "chunk_security_guardrail_001",
        "document_id": "doc_security_compliance",
        "content": (
            "Deterministic security guardrails inspect pre-execution prompts for prompt injection, "
            "adversarial jailbreak attempts, and SQL injection patterns. Post-execution filters sanitize "
            "PII including email addresses and credit card numbers."
        ),
        "metadata": {
            "source": "security_compliance_policy.pdf",
            "department": "Security & Compliance",
            "title": "Enterprise Security Guardrails & Compliance Policy",
            "category": "Security & Privacy",
            "version": "3.1",
            "author": "InfoSec Compliance Team"
        }
    },
    {
        "id": "chunk_security_guardrail_002",
        "document_id": "doc_security_compliance",
        "content": (
            "Role-Based Access Control (RBAC) policy: Data access boundaries must be strictly partitioned by "
            "department. Unauthenticated access or cross-department permission escalation must be rejected immediately."
        ),
        "metadata": {
            "source": "security_compliance_policy.pdf",
            "department": "Security & Compliance",
            "title": "Enterprise Security Guardrails & Compliance Policy",
            "category": "Access Control",
            "version": "3.1",
            "author": "InfoSec Compliance Team"
        }
    }
]


class PGVectorStore:
    """
    Enterprise Dual-Mode Vector Store.
    
    Supports:
    - PostgreSQL + PGVector via SQLAlchemy cosine_distance when database is live.
    - Automatic In-Memory fallback using SIMD/NumPy vector operations or pure Python vector math.
    - Pre-populated enterprise documentation chunks.
    - Metadata & department filtering.
    - Reciprocal Rank Fusion (RRF) hybrid search.
    - Document deletion and summary aggregation.
    """

    def __init__(
        self,
        mode: str = "auto",
        similarity_threshold: Optional[float] = None,
        session_factory=None,
        seed_defaults: bool = True
    ):
        self.mode = mode.lower()  # "auto", "postgres", "in_memory"
        self.threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        self.session_factory = session_factory if session_factory is not None else AsyncSessionLocal
        self.embedding_service = EmbeddingService()
        self._in_memory_chunks: Dict[str, Dict[str, Any]] = {}
        self._postgres_available: Optional[bool] = None

        if seed_defaults:
            self._seed_in_memory_defaults()

    def _seed_in_memory_defaults(self):
        """Populates in-memory storage with realistic enterprise documentation chunks."""
        for doc in DEFAULT_ENTERPRISE_DOCUMENTS:
            embedding = doc.get("embedding")
            if embedding is None:
                embedding = self.embedding_service._generate_vector(doc["content"])
            self._in_memory_chunks[doc["id"]] = {
                "id": doc["id"],
                "document_id": doc["document_id"],
                "content": doc["content"],
                "embedding": embedding,
                "metadata": dict(doc.get("metadata", {}))
            }

    async def init_db(self) -> bool:
        """Initializes database schema and pgvector extension if available."""
        if self.mode == "in_memory" or not HAS_SQLALCHEMY or not self.session_factory or not engine:
            self._postgres_available = False
            return False

        try:
            async with engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.run_sync(Base.metadata.create_all)
            self._postgres_available = True
            return True
        except Exception:
            self._postgres_available = False
            return False

    def is_postgres_active(self) -> bool:
        """Determines if PostgreSQL backend is currently usable."""
        if self.mode == "in_memory":
            return False
        if self.mode == "postgres":
            return True if self._postgres_available is not False else False
        # In 'auto' mode
        if not HAS_SQLALCHEMY or not self.session_factory:
            return False
        return bool(self._postgres_available)

    async def insert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Inserts document chunks into vector store.
        Falls back to in-memory store if PostgreSQL is unavailable.
        """
        if not chunks:
            return 0

        # Always update in-memory cache
        for c in chunks:
            cid = str(c.get("id") or str(uuid.uuid4()))
            doc_id = str(c.get("document_id") or "default_doc")
            content = str(c.get("content", ""))
            embedding = c.get("embedding")
            if embedding is None:
                embedding = self.embedding_service._generate_vector(content)
            
            self._in_memory_chunks[cid] = {
                "id": cid,
                "document_id": doc_id,
                "content": content,
                "embedding": embedding,
                "metadata": dict(c.get("metadata", {}))
            }

        # Attempt PostgreSQL insert if active
        if self.is_postgres_active() and HAS_SQLALCHEMY and DocumentChunk is not None:
            try:
                async with self.session_factory() as session:
                    for c in chunks:
                        cid = str(c.get("id") or str(uuid.uuid4()))
                        doc_id = str(c.get("document_id") or "default_doc")
                        content = str(c.get("content", ""))
                        embedding = c.get("embedding")
                        if embedding is None:
                            embedding = self.embedding_service._generate_vector(content)
                        record = DocumentChunk(
                            id=cid,
                            document_id=doc_id,
                            content=content,
                            embedding=embedding,
                            metadata_=c.get("metadata", {})
                        )
                        await session.merge(record)
                    await session.commit()
            except Exception:
                self._postgres_available = False

        return len(chunks)

    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        department: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Evaluates whether chunk metadata satisfies department and metadata filters."""
        if department is not None:
            chunk_dept = str(metadata.get("department", "")).strip().lower()
            target_dept = str(department).strip().lower()
            if chunk_dept != target_dept:
                return False

        if metadata_filter:
            for key, val in metadata_filter.items():
                if key not in metadata:
                    return False
                if isinstance(val, str) and isinstance(metadata[key], str):
                    if metadata[key].strip().lower() != val.strip().lower():
                        return False
                elif metadata[key] != val:
                    return False

        return True

    async def search(
        self,
        query_vector: List[float],
        limit: int = 4,
        department: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs vector cosine similarity search.
        Falls back to in-memory SIMD/NumPy search if PostgreSQL is unavailable.
        """
        min_sim = threshold if threshold is not None else 0.0

        if self.is_postgres_active() and HAS_SQLALCHEMY and DocumentChunk is not None:
            try:
                async with self.session_factory() as session:
                    distance = DocumentChunk.embedding.cosine_distance(query_vector)
                    stmt = select(
                        DocumentChunk,
                        (1.0 - distance).label("similarity")
                    ).order_by(distance)

                    if department is not None:
                        stmt = stmt.where(DocumentChunk.metadata_["department"].as_string() == department)

                    if metadata_filter:
                        for k, v in metadata_filter.items():
                            stmt = stmt.where(DocumentChunk.metadata_[k].as_string() == str(v))

                    result = await session.execute(stmt)
                    rows = result.all()

                    results = []
                    for chunk, sim in rows:
                        sim_float = float(sim)
                        if threshold is not None and sim_float < threshold:
                            continue
                        results.append({
                            "id": chunk.id,
                            "document_id": chunk.document_id,
                            "content": chunk.content,
                            "similarity": round(sim_float, 4),
                            "metadata": chunk.metadata_ or {}
                        })
                        if len(results) >= limit:
                            break
                    return results
            except Exception:
                self._postgres_available = False

        # In-Memory Search Fallback
        return self._in_memory_search(
            query_vector=query_vector,
            limit=limit,
            department=department,
            metadata_filter=metadata_filter,
            threshold=threshold
        )

    def _in_memory_search(
        self,
        query_vector: List[float],
        limit: int = 4,
        department: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Internal in-memory cosine similarity search."""
        if not self._in_memory_chunks:
            return []

        candidates = []
        for chunk in self._in_memory_chunks.values():
            meta = chunk.get("metadata", {})
            if not self._matches_filters(meta, department=department, metadata_filter=metadata_filter):
                continue

            sim = EmbeddingService.cosine_similarity(query_vector, chunk["embedding"])
            if threshold is not None and sim < threshold:
                continue

            candidates.append({
                "id": chunk["id"],
                "document_id": chunk["document_id"],
                "content": chunk["content"],
                "similarity": round(float(sim), 4),
                "metadata": meta
            })

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:limit]

    @staticmethod
    def _lexical_score(query_text: str, content: str) -> float:
        """Computes lexical relevance score for text search."""
        if not query_text or not content:
            return 0.0

        query_tokens = [t.lower() for t in re.findall(r"\w+", query_text) if len(t) > 1]
        if not query_tokens:
            return 0.0

        content_lower = content.lower()
        content_tokens = re.findall(r"\w+", content_lower)
        if not content_tokens:
            return 0.0

        content_token_set = set(content_tokens)
        token_count = len(content_tokens)

        matched = 0
        freq_sum = 0
        for token in query_tokens:
            if token in content_token_set:
                matched += 1
                freq_sum += content_tokens.count(token)

        # Base token overlap score
        overlap_score = matched / len(query_tokens)
        freq_density = freq_sum / math.sqrt(token_count)
        base_score = (overlap_score * 0.7) + (min(freq_density, 1.0) * 0.3)

        # Exact phrase bonus
        if query_text.lower() in content_lower:
            base_score += 1.5

        return base_score

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        limit: int = 4,
        department: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        rrf_k: int = 60,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Executes Reciprocal Rank Fusion (RRF) Hybrid Search combining:
        1. Dense semantic vector similarity
        2. Lexical keyword relevance
        
        RRF Score Formula:
            RRF(d) = alpha / (rrf_k + rank_vector(d)) + (1 - alpha) / (rrf_k + rank_lexical(d))
        """
        if query_vector is None:
            query_vector = await self.embedding_service.get_embedding(query_text)

        # Filter candidate pool
        filtered_chunks = [
            c for c in self._in_memory_chunks.values()
            if self._matches_filters(c.get("metadata", {}), department=department, metadata_filter=metadata_filter)
        ]

        if not filtered_chunks:
            return []

        # 1. Vector Ranking
        vector_scored = []
        for chunk in filtered_chunks:
            sim = EmbeddingService.cosine_similarity(query_vector, chunk["embedding"])
            vector_scored.append((chunk, float(sim)))

        vector_scored.sort(key=lambda x: x[1], reverse=True)
        vector_ranks = {chunk["id"]: rank + 1 for rank, (chunk, _) in enumerate(vector_scored)}
        vector_sim_map = {chunk["id"]: sim for chunk, sim in vector_scored}

        # 2. Lexical Ranking
        lexical_scored = []
        for chunk in filtered_chunks:
            lscore = self._lexical_score(query_text, chunk["content"])
            lexical_scored.append((chunk, float(lscore)))

        lexical_scored.sort(key=lambda x: x[1], reverse=True)
        lexical_ranks = {chunk["id"]: rank + 1 for rank, (chunk, _) in enumerate(lexical_scored)}
        lexical_score_map = {chunk["id"]: score for chunk, score in lexical_scored}

        # 3. Reciprocal Rank Fusion
        fused_results = []

        for chunk in filtered_chunks:
            cid = chunk["id"]
            vrank = vector_ranks.get(cid, len(filtered_chunks) + 1)
            lrank = lexical_ranks.get(cid, len(filtered_chunks) + 1)

            v_sim = vector_sim_map.get(cid, 0.0)
            if threshold is not None and v_sim < threshold:
                continue

            # RRF calculation
            rrf_score = (alpha / (rrf_k + vrank)) + ((1.0 - alpha) / (rrf_k + lrank))

            fused_results.append({
                "id": cid,
                "document_id": chunk["document_id"],
                "content": chunk["content"],
                "similarity": round(v_sim, 4),
                "rrf_score": round(rrf_score, 6),
                "lexical_score": round(lexical_score_map.get(cid, 0.0), 4),
                "metadata": chunk.get("metadata", {})
            })

        # Sort by RRF score descending, tiebreaker on vector similarity
        fused_results.sort(key=lambda x: (x["rrf_score"], x["similarity"]), reverse=True)
        return fused_results[:limit]

    async def delete_document(self, document_id: str) -> int:
        """
        Deletes all chunks associated with a specific document_id.
        Returns the number of deleted chunks.
        """
        target_doc_id = str(document_id)
        
        # Delete from in-memory store
        to_delete = [
            cid for cid, chunk in self._in_memory_chunks.items()
            if str(chunk.get("document_id")) == target_doc_id
        ]
        for cid in to_delete:
            del self._in_memory_chunks[cid]

        # Delete from PostgreSQL if active
        if self.is_postgres_active() and HAS_SQLALCHEMY and DocumentChunk is not None:
            try:
                async with self.session_factory() as session:
                    stmt = delete(DocumentChunk).where(DocumentChunk.document_id == target_doc_id)
                    res = await session.execute(stmt)
                    await session.commit()
                    return max(len(to_delete), res.rowcount if hasattr(res, 'rowcount') else len(to_delete))
            except Exception:
                self._postgres_available = False

        return len(to_delete)

    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Returns summary records of all indexed documents with chunk counts.
        """
        docs_summary: Dict[str, Dict[str, Any]] = {}

        # In PostgreSQL mode if active
        if self.is_postgres_active() and HAS_SQLALCHEMY and DocumentChunk is not None:
            try:
                async with self.session_factory() as session:
                    stmt = select(
                        DocumentChunk.document_id,
                        func.count(DocumentChunk.id).label("chunk_count")
                    ).group_by(DocumentChunk.document_id)
                    res = await session.execute(stmt)
                    db_docs = res.all()
                    
                    results = []
                    for doc_id, count in db_docs:
                        results.append({
                            "document_id": doc_id,
                            "chunk_count": count,
                            "title": doc_id,
                            "metadata": {}
                        })
                    return results
            except Exception:
                self._postgres_available = False

        # In-Memory aggregation
        for chunk in self._in_memory_chunks.values():
            doc_id = chunk["document_id"]
            meta = chunk.get("metadata", {})
            title = meta.get("title") or meta.get("source") or doc_id
            department = meta.get("department", "")

            if doc_id not in docs_summary:
                docs_summary[doc_id] = {
                    "document_id": doc_id,
                    "chunk_count": 0,
                    "title": title,
                    "department": department,
                    "metadata": meta
                }
            docs_summary[doc_id]["chunk_count"] += 1

        return list(docs_summary.values())

