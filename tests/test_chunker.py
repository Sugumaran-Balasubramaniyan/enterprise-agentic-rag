import unittest
from app.rag.chunker import RecursiveSemanticChunker

class TestChunker(unittest.TestCase):
    def test_chunker_basic(self):
        chunker = RecursiveSemanticChunker(chunk_size=35, chunk_overlap=10)
        text = "Paragraph one with enterprise context.\n\nParagraph two with additional details.\n\nParagraph three."
        chunks = chunker.chunk_text(text, metadata={"doc_id": "test_1"})
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["metadata"]["doc_id"], "test_1")
        self.assertIn("chunk_index", chunks[0]["metadata"])

if __name__ == "__main__":
    unittest.main()
