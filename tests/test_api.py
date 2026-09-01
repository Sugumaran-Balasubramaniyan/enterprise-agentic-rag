import unittest

class TestApiModule(unittest.TestCase):
    def test_schemas_definition(self):
        from app.api.schemas import (
            QueryRequest, QueryResponse,
            DocumentSummaryResponse, DocumentDetailResponse,
            DeleteDocumentResponse, SystemMetricsResponse
        )
        req = QueryRequest(query="What is HNSW?")
        self.assertEqual(req.query, "What is HNSW?")
        self.assertEqual(req.user_role, "standard_user")

        doc_sum = DocumentSummaryResponse(
            document_id="doc_1",
            title="Arch Standard",
            department="Platform Engineering",
            chunk_count=3,
            metadata={"version": "1.0"}
        )
        self.assertEqual(doc_sum.document_id, "doc_1")
        self.assertEqual(doc_sum.chunk_count, 3)

        doc_detail = DocumentDetailResponse(
            document_id="doc_1",
            title="Arch Standard",
            total_chunks=1,
            chunks=[{"id": "c1", "content": "hello"}]
        )
        self.assertEqual(doc_detail.total_chunks, 1)

        del_resp = DeleteDocumentResponse(
            document_id="doc_1",
            chunks_deleted=3,
            status="success"
        )
        self.assertEqual(del_resp.chunks_deleted, 3)

        sys_metric = SystemMetricsResponse(
            total_queries=10,
            blocked_queries=2,
            avg_latency_ms=15.5,
            p95_latency_ms=45.0,
            total_documents=5,
            total_chunks=25,
            active_backend="in_memory"
        )
        self.assertEqual(sys_metric.total_queries, 10)
        self.assertEqual(sys_metric.active_backend, "in_memory")

if __name__ == "__main__":
    unittest.main()
