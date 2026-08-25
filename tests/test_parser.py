import unittest
import tempfile
import os
from pathlib import Path
from app.rag.parser import EnterpriseDocumentParser, DocumentType, ParsedDocument
from app.rag.chunker import RecursiveSemanticChunker


class TestEnterpriseDocumentParser(unittest.TestCase):
    def setUp(self):
        self.parser = EnterpriseDocumentParser()

    def test_plain_text_parsing(self):
        text = "This is the first enterprise paragraph.\n\nThis is the second paragraph with policy info."
        doc = self.parser.parse(text, doc_type=DocumentType.TEXT, metadata={"author": "compliance"})
        self.assertIsInstance(doc, ParsedDocument)
        self.assertEqual(doc.doc_type, DocumentType.TEXT)
        self.assertEqual(doc.metadata["format"], "plain_text")
        self.assertEqual(doc.metadata["paragraph_count"], 2)
        self.assertEqual(doc.metadata["author"], "compliance")
        self.assertEqual(len(doc.sections), 2)
        self.assertIn("first enterprise paragraph", doc.content)

    def test_markdown_parsing(self):
        md_text = (
            "# Enterprise Architecture Overview\n"
            "This document outlines the core architecture.\n\n"
            "## Security Boundary\n"
            "Deterministic guardrails enforce zero trust access.\n\n"
            "## Storage Engine\n"
            "PostgreSQL 16 with PGVector HNSW indexing."
        )
        doc = self.parser.parse(md_text, doc_type=DocumentType.MARKDOWN, filename="arch.md")
        self.assertEqual(doc.doc_type, DocumentType.MARKDOWN)
        self.assertEqual(doc.metadata["headings_count"], 3)
        self.assertEqual(doc.metadata["filename"], "arch.md")
        self.assertTrue(any(s["title"] == "Security Boundary" for s in doc.sections))
        self.assertTrue(any(s["title"] == "Storage Engine" for s in doc.sections))

    def test_csv_tabular_parsing(self):
        csv_data = (
            "Service,Region,Latency_ms,Status\n"
            "RAG-Gateway,eu-west-3,14.2,Healthy\n"
            "PGVector-Store,eu-west-3,8.5,Healthy\n"
            "Auth-Shield,eu-west-3,3.1,Healthy"
        )
        doc = self.parser.parse(csv_data, doc_type=DocumentType.CSV)
        self.assertEqual(doc.doc_type, DocumentType.CSV)
        self.assertEqual(doc.metadata["row_count"], 3)
        self.assertEqual(doc.metadata["columns"], ["Service", "Region", "Latency_ms", "Status"])
        self.assertIn("Record 1 -> Service: RAG-Gateway; Region: eu-west-3; Latency_ms: 14.2; Status: Healthy", doc.content)

    def test_tsv_tabular_parsing(self):
        tsv_data = (
            "Metric\tThreshold\tEnforced\n"
            "Similarity\t0.70\tYes\n"
            "Grounding\t0.20\tYes"
        )
        doc = self.parser.parse(tsv_data, doc_type=DocumentType.TSV)
        self.assertEqual(doc.doc_type, DocumentType.TSV)
        self.assertEqual(doc.metadata["row_count"], 2)
        self.assertIn("Record 1 -> Metric: Similarity; Threshold: 0.70; Enforced: Yes", doc.content)

    def test_json_tabular_parsing(self):
        json_data = (
            '[\n'
            '  {"tenant": "Finance", "quota": 10000, "region": "eu-west-3"},\n'
            '  {"tenant": "Engineering", "quota": 50000, "region": "eu-west-3"}\n'
            ']'
        )
        doc = self.parser.parse(json_data, doc_type=DocumentType.JSON_TABLE)
        self.assertEqual(doc.doc_type, DocumentType.JSON_TABLE)
        self.assertEqual(doc.metadata["record_count"], 2)
        self.assertIn("Record 1: tenant: Finance, quota: 10000, region: eu-west-3", doc.content)

    def test_ocr_scanned_parsing_and_remediation(self):
        ocr_scan = (
            "[OCR_CONFIDENCE: 96.4%]\n"
            "The enter-\nprise security policy requires zero trust.\n\n"
            "All model outputs must adhere to deterministic schema vali-\ndation.\n"
            "\x0c"
            "Section 2: High availability and failover protocols."
        )
        doc = self.parser.parse(ocr_scan, doc_type=DocumentType.OCR_SCANNED)
        self.assertEqual(doc.doc_type, DocumentType.OCR_SCANNED)
        self.assertEqual(doc.metadata["ocr_confidence"], 96.4)
        self.assertEqual(doc.metadata["page_count"], 2)
        self.assertTrue(doc.metadata["is_ocr_remediated"])
        # Check hyphen stitching
        self.assertIn("enterprise security policy", doc.content)
        self.assertIn("validation", doc.content)
        self.assertNotIn("enter-\nprise", doc.content)

    def test_auto_detection(self):
        # Auto detect markdown
        md = "# Header 1\nSome text here\n## Subheader\nMore text"
        self.assertEqual(self.parser.detect_doc_type(md), DocumentType.MARKDOWN)

        # Auto detect CSV
        csv_sample = "a,b,c\n1,2,3\n4,5,6"
        self.assertEqual(self.parser.detect_doc_type(csv_sample), DocumentType.CSV)

        # Auto detect JSON
        json_sample = '[{"key": "value"}]'
        self.assertEqual(self.parser.detect_doc_type(json_sample), DocumentType.JSON_TABLE)

        # Auto detect OCR
        ocr_sample = "[OCR_CONFIDENCE: 88.0%]\nScanned text\x0cPage 2 text"
        self.assertEqual(self.parser.detect_doc_type(ocr_sample), DocumentType.OCR_SCANNED)

        # Filename override
        self.assertEqual(self.parser.detect_doc_type("random text", filename="audit.md"), DocumentType.MARKDOWN)
        self.assertEqual(self.parser.detect_doc_type("random text", filename="data.csv"), DocumentType.CSV)

    def test_parse_and_chunk_integration(self):
        chunker = RecursiveSemanticChunker(chunk_size=60, chunk_overlap=10)
        text = (
            "Section 1 details the enterprise security posture.\n\n"
            "Section 2 covers the deterministic guardrails and validation filters.\n\n"
            "Section 3 covers the high-speed HNSW pgvector indices."
        )
        chunks = self.parser.parse_and_chunk(
            text,
            chunker=chunker,
            doc_type=DocumentType.TEXT,
            metadata={"source_system": "compliance_portal"}
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["source_system"], "compliance_portal")
        self.assertEqual(chunks[0]["metadata"]["doc_type"], "text")
        self.assertIn("chunk_index", chunks[0]["metadata"])

    def test_parse_file(self):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as tf:
            tf.write("# Temp Doc\n\nContent inside temporary file.")
            tf_path = tf.name

        try:
            doc = self.parser.parse_file(tf_path)
            self.assertEqual(doc.doc_type, DocumentType.MARKDOWN)
            self.assertIn("Content inside temporary file", doc.content)
            self.assertEqual(doc.metadata["filename"], Path(tf_path).name)
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)


if __name__ == "__main__":
    unittest.main()
