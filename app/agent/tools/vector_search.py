from typing import Dict, Any, Optional, List
from app.rag.vector_store import PGVectorStore
from app.rag.embeddings import EmbeddingService


class VectorSearchTool:
    """
    Enterprise Vector Search Tool.
    Integrates with PGVectorStore supporting dense cosine similarity vector search
    and Reciprocal Rank Fusion (RRF) hybrid search with department/metadata filtering.
    """
    name = "vector_search"
    description = (
        "Searches enterprise knowledge base using PGVector dense semantic similarity "
        "or Reciprocal Rank Fusion (RRF) hybrid search with department filtering."
    )

    def __init__(self, vector_store: Optional[PGVectorStore] = None):
        self.vector_store = vector_store or PGVectorStore()
        self.embedding_service = EmbeddingService()

    async def execute(
        self,
        query: str,
        limit: int = 3,
        department: Optional[str] = None,
        use_hybrid: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes vector search or hybrid search over indexed enterprise documentation.
        """
        query_clean = query.strip() if query else ""
        if not query_clean:
            return {
                "results": [],
                "total_found": 0,
                "search_type": "hybrid_rrf" if use_hybrid else "dense_vector",
                "department_filter": department,
                "query": query
            }

        if use_hybrid:
            query_vector = await self.embedding_service.get_embedding(query_clean)
            results = await self.vector_store.hybrid_search(
                query_text=query_clean,
                query_vector=query_vector,
                limit=limit,
                department=department,
                metadata_filter=metadata_filter,
                threshold=threshold
            )
            search_type = "hybrid_rrf"
        else:
            query_vector = await self.embedding_service.get_embedding(query_clean)
            results = await self.vector_store.search(
                query_vector=query_vector,
                limit=limit,
                department=department,
                metadata_filter=metadata_filter,
                threshold=threshold
            )
            search_type = "dense_vector"

        return {
            "results": results,
            "total_found": len(results),
            "search_type": search_type,
            "department_filter": department,
            "query": query_clean
        }
