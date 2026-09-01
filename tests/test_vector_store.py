import unittest
import asyncio
import math
from app.rag.vector_store import PGVectorStore, DEFAULT_ENTERPRISE_DOCUMENTS
from app.rag.embeddings import EmbeddingService


class TestPGVectorStore(unittest.TestCase):
    def setUp(self):
        self.embedding_service = EmbeddingService()
        self.store = PGVectorStore(mode="in_memory", seed_defaults=True)

    def test_embeddings_normalization_and_batch(self):
        async def run():
            embedder = EmbeddingService()
            vec = await embedder.get_embedding("Deterministic enterprise guardrails")
            self.assertEqual(len(vec), embedder.dimension)
            
            # Check Euclidean norm is 1.0 (unit vector)
            norm = math.sqrt(sum(x * x for x in vec))
            self.assertAlmostEqual(norm, 1.0, places=5)

            # Test batch embedding
            texts = [
                "Platform Engineering architecture standards",
                "PGVector HNSW indexing guide",
                "Security compliance and PII sanitization"
            ]
            batch_vecs = await embedder.get_embeddings_batch(texts)
            self.assertEqual(len(batch_vecs), 3)
            for b_vec in batch_vecs:
                self.assertEqual(len(b_vec), embedder.dimension)
                b_norm = math.sqrt(sum(x * x for x in b_vec))
                self.assertAlmostEqual(b_norm, 1.0, places=5)

            # Test cosine similarity static method
            sim_identical = EmbeddingService.cosine_similarity(vec, vec)
            self.assertAlmostEqual(sim_identical, 1.0, places=5)

            # Opposite vector
            neg_vec = [-x for x in vec]
            sim_opposite = EmbeddingService.cosine_similarity(vec, neg_vec)
            self.assertAlmostEqual(sim_opposite, -1.0, places=5)

        asyncio.run(run())

    def test_default_enterprise_chunks_prepopulated(self):
        async def run():
            store = PGVectorStore(mode="in_memory", seed_defaults=True)
            all_docs = await store.get_all_documents()
            self.assertGreaterEqual(len(all_docs), 3)

            doc_ids = {d["document_id"] for d in all_docs}
            self.assertIn("doc_arch_standards", doc_ids)
            self.assertIn("doc_pgvector_hnsw", doc_ids)
            self.assertIn("doc_security_compliance", doc_ids)

            # Search default content
            query_vec = await self.embedding_service.get_embedding(
                "PGVector HNSW indexes offer sub-20ms retrieval"
            )
            results = await store.search(query_vec, limit=2)
            self.assertGreater(len(results), 0)
            self.assertIn("HNSW", results[0]["content"])
            self.assertGreater(results[0]["similarity"], 0.45)

        asyncio.run(run())

    def test_vector_cosine_similarity_search_in_memory(self):
        async def run():
            store = PGVectorStore(mode="in_memory", seed_defaults=False)

            # Create test vectors with known relationships
            dim = store.embedding_service.dimension
            
            # Base target vector (unit vector along first dimension)
            v_target = [0.0] * dim
            v_target[0] = 1.0

            # Very close vector (small angle)
            v_close = [0.0] * dim
            v_close[0] = 0.95
            v_close[1] = math.sqrt(1.0 - 0.95**2)

            # Distant vector (orthogonal)
            v_distant = [0.0] * dim
            v_distant[1] = 1.0

            # Opposite vector
            v_opposite = [0.0] * dim
            v_opposite[0] = -1.0

            chunks = [
                {
                    "id": "chunk_target",
                    "document_id": "doc_test_1",
                    "content": "Target reference document content.",
                    "embedding": v_target,
                    "metadata": {"category": "target", "department": "Platform Engineering"}
                },
                {
                    "id": "chunk_close",
                    "document_id": "doc_test_1",
                    "content": "Close semantic document content.",
                    "embedding": v_close,
                    "metadata": {"category": "close", "department": "Platform Engineering"}
                },
                {
                    "id": "chunk_distant",
                    "document_id": "doc_test_2",
                    "content": "Distant orthogonal document content.",
                    "embedding": v_distant,
                    "metadata": {"category": "distant", "department": "Data Engineering"}
                },
                {
                    "id": "chunk_opposite",
                    "document_id": "doc_test_3",
                    "content": "Opposite polarity document content.",
                    "embedding": v_opposite,
                    "metadata": {"category": "opposite", "department": "Security & Compliance"}
                }
            ]

            await store.insert_chunks(chunks)

            # Search with v_target as query
            results = await store.search(v_target, limit=4)
            self.assertEqual(len(results), 4)

            # Verify closer vectors receive higher similarity scores
            self.assertEqual(results[0]["id"], "chunk_target")
            self.assertAlmostEqual(results[0]["similarity"], 1.0, places=3)

            self.assertEqual(results[1]["id"], "chunk_close")
            self.assertAlmostEqual(results[1]["similarity"], 0.95, places=2)

            self.assertEqual(results[2]["id"], "chunk_distant")
            self.assertAlmostEqual(results[2]["similarity"], 0.0, places=2)

            self.assertEqual(results[3]["id"], "chunk_opposite")
            self.assertAlmostEqual(results[3]["similarity"], -1.0, places=2)

            # Test similarity ordering strictly monotonic descending
            similarities = [r["similarity"] for r in results]
            self.assertEqual(similarities, sorted(similarities, reverse=True))

            # Test limit parameter
            top_2 = await store.search(v_target, limit=2)
            self.assertEqual(len(top_2), 2)
            self.assertEqual(top_2[0]["id"], "chunk_target")
            self.assertEqual(top_2[1]["id"], "chunk_close")

            # Test similarity threshold filtering
            high_threshold_results = await store.search(v_target, threshold=0.90)
            self.assertEqual(len(high_threshold_results), 2)
            for r in high_threshold_results:
                self.assertGreaterEqual(r["similarity"], 0.90)

        asyncio.run(run())

    def test_metadata_and_department_filtering(self):
        async def run():
            store = PGVectorStore(mode="in_memory", seed_defaults=False)
            
            vec = await self.embedding_service.get_embedding("microservices policy")

            chunks = [
                {
                    "id": "c1",
                    "document_id": "doc_platform",
                    "content": "Platform Engineering standards and guidelines.",
                    "embedding": vec,
                    "metadata": {"department": "Platform Engineering", "env": "prod", "tier": 1}
                },
                {
                    "id": "c2",
                    "document_id": "doc_data",
                    "content": "Data Engineering streaming pipelines.",
                    "embedding": vec,
                    "metadata": {"department": "Data Engineering", "env": "prod", "tier": 2}
                },
                {
                    "id": "c3",
                    "document_id": "doc_sec",
                    "content": "Security & Compliance audit checklist.",
                    "embedding": vec,
                    "metadata": {"department": "Security & Compliance", "env": "staging", "tier": 1}
                }
            ]

            await store.insert_chunks(chunks)

            # Department filter
            platform_results = await store.search(vec, department="Platform Engineering")
            self.assertEqual(len(platform_results), 1)
            self.assertEqual(platform_results[0]["id"], "c1")
            self.assertEqual(platform_results[0]["metadata"]["department"], "Platform Engineering")

            # Case-insensitive department filter
            data_results = await store.search(vec, department="data engineering")
            self.assertEqual(len(data_results), 1)
            self.assertEqual(data_results[0]["id"], "c2")

            # Metadata dict filter
            prod_tier1 = await store.search(vec, metadata_filter={"env": "prod", "tier": 1})
            self.assertEqual(len(prod_tier1), 1)
            self.assertEqual(prod_tier1[0]["id"], "c1")

            staging_results = await store.search(vec, metadata_filter={"env": "staging"})
            self.assertEqual(len(staging_results), 1)
            self.assertEqual(staging_results[0]["id"], "c3")

            # Non-matching filter returns empty
            empty_results = await store.search(vec, department="NonExistentDept")
            self.assertEqual(len(empty_results), 0)

        asyncio.run(run())

    def test_hybrid_search_rrf(self):
        async def run():
            store = PGVectorStore(mode="in_memory", seed_defaults=False)

            c1_content = "PGVector HNSW indexing delivers sub-20ms latency for vector similarity queries."
            c2_content = "Zero downtime failover across multiple availability zones in eu-west-3."
            c3_content = "Deterministic security guardrails filter prompt injections and sanitize PII."

            v1 = await self.embedding_service.get_embedding(c1_content)
            v2 = await self.embedding_service.get_embedding(c2_content)
            v3 = await self.embedding_service.get_embedding(c3_content)

            chunks = [
                {
                    "id": "chunk_pg_hnsw",
                    "document_id": "doc_db",
                    "content": c1_content,
                    "embedding": v1,
                    "metadata": {"department": "Data Engineering", "topic": "vector_indexing"}
                },
                {
                    "id": "chunk_cloud_resilience",
                    "document_id": "doc_cloud",
                    "content": c2_content,
                    "embedding": v2,
                    "metadata": {"department": "Platform Engineering", "topic": "high_availability"}
                },
                {
                    "id": "chunk_guardrails",
                    "document_id": "doc_sec",
                    "content": c3_content,
                    "embedding": v3,
                    "metadata": {"department": "Security & Compliance", "topic": "safety"}
                }
            ]

            await store.insert_chunks(chunks)

            # Query with specific keywords and vector
            query_text = "PGVector HNSW latency sub-20ms"
            query_vector = await self.embedding_service.get_embedding(query_text)

            results = await store.hybrid_search(
                query_text=query_text,
                query_vector=query_vector,
                limit=3,
                rrf_k=60,
                alpha=0.5
            )

            self.assertGreater(len(results), 0)
            
            # The exact match should be ranked #1
            top_match = results[0]
            self.assertEqual(top_match["id"], "chunk_pg_hnsw")
            self.assertIn("rrf_score", top_match)
            self.assertIn("similarity", top_match)
            self.assertIn("lexical_score", top_match)
            self.assertGreater(top_match["rrf_score"], 0.0)
            self.assertGreater(top_match["lexical_score"], 0.0)

            # Verify RRF sorting order
            rrf_scores = [r["rrf_score"] for r in results]
            self.assertEqual(rrf_scores, sorted(rrf_scores, reverse=True))

            # Hybrid search with department filter
            sec_hybrid = await store.hybrid_search(
                query_text="guardrails filter PII",
                department="Security & Compliance"
            )
            self.assertEqual(len(sec_hybrid), 1)
            self.assertEqual(sec_hybrid[0]["id"], "chunk_guardrails")

        asyncio.run(run())

    def test_delete_document(self):
        async def run():
            store = PGVectorStore(mode="in_memory", seed_defaults=False)

            # Insert document with 3 chunks
            doc_id = "doc_to_delete_123"
            chunks = [
                {
                    "id": f"chunk_del_{i}",
                    "document_id": doc_id,
                    "content": f"Chunk {i} of transient document.",
                    "embedding": [0.1] * store.embedding_service.dimension,
                    "metadata": {"doc_index": i}
                }
                for i in range(3)
            ]

            # Also insert another persistent document
            persistent_chunks = [
                {
                    "id": "chunk_keep_1",
                    "document_id": "doc_keep_456",
                    "content": "Keep this document intact.",
                    "embedding": [0.2] * store.embedding_service.dimension,
                    "metadata": {"status": "permanent"}
                }
            ]

            await store.insert_chunks(chunks)
            await store.insert_chunks(persistent_chunks)

            # Verify docs before deletion
            docs = await store.get_all_documents()
            doc_map = {d["document_id"]: d["chunk_count"] for d in docs}
            self.assertEqual(doc_map.get(doc_id), 3)
            self.assertEqual(doc_map.get("doc_keep_456"), 1)

            # Delete the document
            deleted_count = await store.delete_document(doc_id)
            self.assertEqual(deleted_count, 3)

            # Verify chunks are gone
            docs_after = await store.get_all_documents()
            doc_map_after = {d["document_id"]: d["chunk_count"] for d in docs_after}
            self.assertNotIn(doc_id, doc_map_after)
            self.assertEqual(doc_map_after.get("doc_keep_456"), 1)

            # Search should not return deleted chunks
            search_res = await store.search([0.1] * store.embedding_service.dimension)
            for r in search_res:
                self.assertNotEqual(r["document_id"], doc_id)

            # Deleting non-existent document returns 0
            zero_deleted = await store.delete_document("non_existent_doc")
            self.assertEqual(zero_deleted, 0)

        asyncio.run(run())

    def test_get_all_documents(self):
        async def run():
            store = PGVectorStore(mode="in_memory", seed_defaults=False)

            chunks = [
                {
                    "id": "c1",
                    "document_id": "doc_alpha",
                    "content": "Alpha document chunk 1",
                    "embedding": [0.05] * store.embedding_service.dimension,
                    "metadata": {"title": "Alpha Architecture", "department": "Platform Engineering"}
                },
                {
                    "id": "c2",
                    "document_id": "doc_alpha",
                    "content": "Alpha document chunk 2",
                    "embedding": [0.05] * store.embedding_service.dimension,
                    "metadata": {"title": "Alpha Architecture", "department": "Platform Engineering"}
                },
                {
                    "id": "c3",
                    "document_id": "doc_beta",
                    "content": "Beta document chunk 1",
                    "embedding": [0.08] * store.embedding_service.dimension,
                    "metadata": {"title": "Beta Specifications", "department": "Data Engineering"}
                }
            ]

            await store.insert_chunks(chunks)

            all_docs = await store.get_all_documents()
            self.assertEqual(len(all_docs), 2)

            summary = {d["document_id"]: d for d in all_docs}
            self.assertEqual(summary["doc_alpha"]["chunk_count"], 2)
            self.assertEqual(summary["doc_alpha"]["title"], "Alpha Architecture")
            self.assertEqual(summary["doc_alpha"]["department"], "Platform Engineering")

            self.assertEqual(summary["doc_beta"]["chunk_count"], 1)
            self.assertEqual(summary["doc_beta"]["title"], "Beta Specifications")
            self.assertEqual(summary["doc_beta"]["department"], "Data Engineering")

        asyncio.run(run())

    def test_live_mode_fallback_and_error_handling(self):
        async def run():
            # In auto mode, when postgres is not reachable, gracefully fallback to in-memory
            store = PGVectorStore(mode="auto", seed_defaults=True)
            
            # init_db should gracefully return False and fallback
            db_ok = await store.init_db()
            self.assertFalse(db_ok)
            self.assertFalse(store.is_postgres_active())

            # Search still succeeds via in-memory fallback
            vec = await self.embedding_service.get_embedding("zero trust policy")
            results = await store.search(vec, limit=2)
            self.assertGreater(len(results), 0)

            # Ingestion still succeeds via in-memory fallback
            new_chunk = [{
                "id": "fallback_test_1",
                "document_id": "doc_fallback",
                "content": "Fallback test content when DB is offline.",
                "embedding": vec,
                "metadata": {"department": "Platform Engineering"}
            }]
            count = await store.insert_chunks(new_chunk)
            self.assertEqual(count, 1)

            # Verify retrieval
            res = await store.search(vec, limit=1)
            self.assertEqual(res[0]["id"], "fallback_test_1")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
