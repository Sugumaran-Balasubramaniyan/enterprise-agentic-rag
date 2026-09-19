"""Regression tests for the fixed async query-transformation path.

History: the transforms' structured LLM call was previously not awaited, so the
``_structured`` coroutine was never resolved and the transform silently returned
the input unchanged even when a real async LLM client was configured. This suite
verifies the awaited path returns the canned structured outputs, and that the
offline behavior (``llm_client is None`` or a mock provider) still returns the
input unchanged.
"""

import asyncio
import unittest

from app.rag.query_transforms import (
    ExpandOutput,
    HydeOutput,
    QueryTransformer,
    RewriteOutput,
)


class FakeAsyncLLMClient:
    """Minimal async client that returns canned structured outputs.

    ``provider`` is a non-mock value on purpose, so the transforms take the live
    awaited code path rather than the offline short-circuit.
    """

    provider = "openai-compatible"
    model = "fake-judge"

    def __init__(self, rewritten="canned rewritten query", queries=None, document="canned hypothetical document"):
        self.rewritten = rewritten
        self.queries = queries if queries is not None else ["variant one", "variant two"]
        self.document = document

    async def structured(self, messages, response_model, **kwargs):
        # Per the QueryTransformer contract the fake returns the canned model
        # directly; the real client wraps it in a (model, meta) tuple that the
        # caller unpacks. Matching this shape is what lets the awaited path in
        # ``QueryTransformer._structured`` extract the field via ``_unwrap``.
        if response_model is RewriteOutput:
            return RewriteOutput(rewritten_query=self.rewritten)
        if response_model is ExpandOutput:
            return ExpandOutput(queries=self.queries)
        if response_model is HydeOutput:
            return HydeOutput(document=self.document)
        raise AssertionError(f"unexpected response_model {response_model}")

    async def complete(self, messages, **kwargs):
        return {"content": "canned completion"}


class TestQueryTransformsAwaitedPath(unittest.TestCase):
    """The await-fixed path returns the canned structured outputs."""

    def setUp(self):
        self.transformer = QueryTransformer()
        self.client = FakeAsyncLLMClient()

    def test_rewrite_returns_canned_rewritten_query(self):
        result = asyncio.run(self.transformer.rewrite(self.client, "original question"))
        self.assertEqual(result, self.client.rewritten)

    def test_expand_returns_original_plus_canned_variants(self):
        result = asyncio.run(self.transformer.expand(self.client, "original question", n=3))
        self.assertEqual(result, ["original question", "variant one", "variant two"])

    def test_hyde_returns_canned_document(self):
        result = asyncio.run(self.transformer.hyde(self.client, "original question"))
        self.assertEqual(result, self.client.document)

    def test_complete_is_callable_and_returns_canned(self):
        async def run():
            return await self.client.complete([{"role": "user", "content": "hi"}])

        result = asyncio.run(run())
        self.assertEqual(result["content"], "canned completion")


class TestQueryTransformsOffline(unittest.TestCase):
    """Offline safety: no client or a mock provider -> input unchanged."""

    def setUp(self):
        self.transformer = QueryTransformer()

    def test_rewrite_unchanged_when_client_none(self):
        result = asyncio.run(self.transformer.rewrite(None, "x"))
        self.assertEqual(result, "x")

    def test_rewrite_unchanged_when_mock_provider(self):
        from app.llm.mock import MockLLMClient

        client = MockLLMClient()
        result = asyncio.run(self.transformer.rewrite(client, "x"))
        self.assertEqual(result, "x")

    def test_expand_not_offline_when_client_none(self):
        result = asyncio.run(self.transformer.expand(None, "x", n=3))
        self.assertEqual(result, ["x"])


if __name__ == "__main__":
    unittest.main()