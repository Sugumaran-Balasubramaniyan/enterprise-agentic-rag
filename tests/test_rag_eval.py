"""Regression tests for the RAG evaluation harness and its golden QA dataset.

Covers:
1. ``data/eval/golden_qa.jsonl`` parses as 20 valid JSON rows with the expected
   keys, a valid ``question_type``, and the required question-type mix.
2. Every ``relevant_chunk_id`` referenced by the golden set actually exists in
   ``DEFAULT_ENTERPRISE_DOCUMENTS`` (the seeded corpus it must be grounded in).
3. ``benchmarks/rag_eval.py --provider mock`` is deterministic and exits 0 while
   printing the ``hit@k`` metric -- the CI gate contract (mock mode needs no
   API keys).
"""

import json
import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLDEN_PATH = os.path.join(REPO_ROOT, "data", "eval", "golden_qa.jsonl")
RAG_EVAL = os.path.join(REPO_ROOT, "benchmarks", "rag_eval.py")

REQUIRED_KEYS = {"id", "question", "reference_answer", "relevant_chunk_ids", "question_type"}
VALID_TYPES = {"factoid", "multi-hop", "calculation", "technical"}


class TestGoldenDataset(unittest.TestCase):
    """Validates the golden QA dataset shape and grounding."""

    @classmethod
    def setUpClass(cls):
        cls.rows = []
        with open(GOLDEN_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cls.rows.append(json.loads(line))

    def test_golden_parses_to_20_valid_rows(self):
        self.assertEqual(len(self.rows), 20, "golden set must contain exactly 20 rows")
        for row in self.rows:
            missing = REQUIRED_KEYS - set(row.keys())
            self.assertEqual(missing, set(), f"row {row.get('id')} missing keys {missing}")

    def test_question_types_valid(self):
        for row in self.rows:
            self.assertIn(row["question_type"], VALID_TYPES, f"row {row['id']} bad type")
        types = [row["question_type"] for row in self.rows]
        for expected in VALID_TYPES:
            self.assertIn(expected, types, f"missing question_type {expected}")
        self.assertGreaterEqual(types.count("factoid"), 8)
        self.assertGreaterEqual(types.count("technical"), 4)
        self.assertGreaterEqual(types.count("multi-hop"), 4)
        self.assertGreaterEqual(types.count("calculation"), 4)

    def test_relevant_chunk_ids_exist(self):
        from app.rag.vector_store import DEFAULT_ENTERPRISE_DOCUMENTS

        known_ids = {doc["id"] for doc in DEFAULT_ENTERPRISE_DOCUMENTS}
        self.assertGreaterEqual(len(known_ids), 6, "seeded corpus must be populated")
        for row in self.rows:
            self.assertIsInstance(row["relevant_chunk_ids"], list)
            self.assertGreater(len(row["relevant_chunk_ids"]), 0, f"row {row['id']} has no chunks")
            for cid in row["relevant_chunk_ids"]:
                self.assertIn(
                    cid,
                    known_ids,
                    f"row {row['id']} references unknown chunk {cid!r}",
                )

    def test_ids_unique(self):
        ids = [row["id"] for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)), "row ids must be unique")


class TestRagEvalMockMode(unittest.TestCase):
    """CI gate contract: mock mode runs offline and exits 0 deterministically."""

    def run_harness(self):
        return subprocess.run(
            [sys.executable, RAG_EVAL, "--provider", "mock"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
        )

    def test_mock_mode_exits_zero_and_prints_hit_at_k(self):
        result = self.run_harness()
        self.assertEqual(
            result.returncode,
            0,
            f"harness exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("hit@k", result.stdout)

    def test_mock_mode_is_deterministic(self):
        first = self.run_harness()
        second = self.run_harness()
        self.assertEqual(first.stdout, second.stdout, "mock mode must be byte-deterministic")


if __name__ == "__main__":
    unittest.main()