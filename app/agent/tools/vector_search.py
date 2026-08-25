from app.rag.vector_store import PGVectorStore
from app.rag.embeddings import EmbeddingService
from typing import Dict, Any

class VectorSearchTool:
    name = "vector_search"
    description = "Searches enterprise knowledge base using PGVector semantic similarity."

    def __init__(self):
        self.vector_store = PGVectorStore()
        self.embedding_service = EmbeddingService()

    async def execute(self, query: str, limit: int = 3) -> Dict[str, Any]:
        query_vector = await self.embedding_service.get_embedding(query)
        results = await self.vector_store.search(query_vector, limit=limit)
        return {
            "results": results,
            "total_found": len(results)
        }
