"""Agent loop tests for the LLM-driven agentic rewrite.

Covers:
(a) Mock-mode behavioral equivalence with the historical loop (the exact queries
    and shape invariants from ``tests/test_orchestrator_loop.py``).
(b) :class:`AdaptiveRouter` deterministic fallback returning ``classify_intent``
    output verbatim.
(c) :class:`RetrievalAssessor` / :class:`ReflectionCritic` offline behavior.
(d) The real-LLM path driven by a canned (network-free) fake client: routing via
    LLM, agentic retrieval (query expansion, rerank), LLM synthesis and
    streaming, all still passing post-execution guardrails.

No network calls are made anywhere in this file.
"""

import asyncio
import re
import unittest
from typing import AsyncIterator, List, Optional

from app.agent.orchestrator import AgentOrchestrator
from app.agent.router import AdaptiveRouter, ReflectionCritic, RetrievalAssessor
from app.llm.client import LLMClient
from app.rag.vector_store import PGVectorStore


# ---------------------------------------------------------------------------
# Network-free fake LLM client
# ---------------------------------------------------------------------------
class FakeLLMClient(LLMClient):
    """Deterministic LLM-client stand-in backing the real-LLM agentic path."""

    provider = "fake"
    model = "fake-gpt"

    def __init__(self):
        self.structured_calls = 0
        self.complete_calls = 0

    @staticmethod
    def _last_user(messages: List[dict]) -> str:
        for msg in reversed(messages or []):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    def _grounded_answer(self, messages: List[dict]) -> str:
        """Answer grounded in the first source chunk embedded in the prompt."""
        text = self._last_user(messages)
        match = re.search(r"\[[^\]]+\]:\s*(.+)", text, re.DOTALL)
        content = match.group(1).strip() if match else text
        return f"Answer: {content} grounded in retrieved documentation."

    async def _chat(
        self,
        messages: List[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> tuple:
        self.complete_calls += 1
        content = self._grounded_answer(messages)
        usage = {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10,
                 "model": self.model, "provider": self.provider}
        return content, None, usage

    async def _stream_chat(
        self,
        messages: List[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        content = self._grounded_answer(messages)
        for word in content.split(" "):
            yield word + " "
        yield ""

    async def structured(self, messages, response_model, *, temperature=None, max_tokens=None):
        """Canned structured outputs keyed off the requested model type."""
        self.structured_calls += 1
        model_name = getattr(response_model, "__name__", "")
        query = self._last_user(messages)

        if model_name == "RouteDecision":
            payload = {
                "needs_retrieval": True,
                "needs_calculation": False,
                "needs_verification": True,
                "department": "Data Engineering",
                "retrieval_mode": "multi",
            }
        elif model_name == "RewriteOutput":
            payload = {"rewritten_query": f"rewritten {query}"}
        elif model_name == "ExpandOutput":
            payload = {"queries": [query, f"variant {query}"]}
        elif model_name == "RetrievalAssessment":
            payload = {"quality": "adequate", "reason": "context is sufficient"}
        elif model_name == "ChunkCritique":
            payload = {"keep_indices": [0, 1, 2]}
        else:
            payload = {}

        parsed = response_model.model_validate(payload)
        meta = {"content": "{}", "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                "total_tokens": 2, "model": self.model, "provider": self.provider}}
        return parsed, meta


def _run(coro):
    """Run a coroutine to completion with a fresh event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        return loop.run_until_complete(coro)
    return asyncio.run(coro)


class TestAgentLoop(unittest.TestCase):

    def setUp(self):
        self.vector_store = PGVectorStore(mode="in_memory", seed_defaults=True)
        # Mock-mode orchestrator (no LLM client -> deterministic path).
        self.mock_orch = AgentOrchestrator(vector_store=self.vector_store, max_steps=5)
        self.fake_llm = FakeLLMClient()

    # ------------------------------------------------------------------
    # (a) Mock-mode equivalence: identical shape to test_orchestrator_loop
    # ------------------------------------------------------------------
    def test_mock_equivalence_retrieval(self):
        async def run():
            resp = await self.mock_orch.execute(
                "What are the platform engineering architecture standards and safety policies?"
            )
            self.assertGreater(len(resp.answer), 20)
            self.assertGreater(len(resp.sources), 0)
            self.assertTrue(resp.guardrail_metrics["pre_execution_passed"])
            self.assertTrue(resp.guardrail_metrics["is_grounded"])
            self.assertGreaterEqual(resp.guardrail_metrics["factual_grounding_score"], 0.20)
            self.assertIn("vector_search", [t.tool_name for t in resp.tool_traces])
            self.assertTrue(any("vector retrieval" in s.lower() for s in resp.reasoning_steps))
            self.assertTrue(any("grounded" in s.lower() for s in resp.reasoning_steps))
            # Mock mode must NOT surface real-LLM agentic metrics.
            self.assertNotIn("retrieval_mode", resp.guardrail_metrics)
            self.assertNotIn("query_variants", resp.guardrail_metrics)
        _run(run())

    def test_mock_equivalence_calculation(self):
        async def run():
            resp = await self.mock_orch.execute(
                "Calculate RAM needed for 5000000 vectors with 1536 dimensions"
            )
            self.assertTrue("30,720,000,000" in resp.answer or "30.72" in resp.answer or "28.61" in resp.answer)
            calc_traces = [t for t in resp.tool_traces if t.tool_name == "calculator"]
            self.assertEqual(len(calc_traces), 1)
            out = calc_traces[0].output
            self.assertEqual(out["formula_type"], "vector_ram")
            self.assertEqual(out["details"]["num_vectors"], 5000000)
            self.assertEqual(out["details"]["dimension"], 1536)
            self.assertEqual(out["details"]["bytes"], 30720000000)
        _run(run())

    def test_mock_equivalence_composite(self):
        async def run():
            resp = await self.mock_orch.execute(
                "What is the PGVector HNSW latency and calculate the concurrency for 500 QPS with 20ms latency?"
            )
            self.assertIn("HNSW", resp.answer)
            self.assertTrue("concurrency" in resp.answer.lower() or "10.0" in resp.answer)
            names = [t.tool_name for t in resp.tool_traces]
            self.assertIn("vector_search", names)
            self.assertIn("calculator", names)
            calc = next(t for t in resp.tool_traces if t.tool_name == "calculator")
            self.assertAlmostEqual(calc.output["result"], 10.0, places=2)
        _run(run())

    # ------------------------------------------------------------------
    # (b) AdaptiveRouter deterministic fallback
    # ------------------------------------------------------------------
    def test_router_mock_fallback_is_verbatim_classify_intent(self):
        orch = self.mock_orch
        router = AdaptiveRouter(classify_fallback=orch.classify_intent)
        for llm in (None, orch._get_llm_client()):  # None and a mock-provider client
            async def run(llm=llm):
                intent = await router.route("Calculate RAM for 5 vectors with 1536 dims", user_role="standard_user", llm_client=llm)
                expected = orch.classify_intent("Calculate RAM for 5 vectors with 1536 dims", user_role="standard_user")
                self.assertEqual(intent, expected)
                self.assertNotIn("retrieval_mode", intent)
            _run(run())

    # ------------------------------------------------------------------
    # (c) Offline retrieval quality gates
    # ------------------------------------------------------------------
    def test_assessor_offline_adequate(self):
        async def run():
            assessor = RetrievalAssessor()
            chunks = [{"id": "c1", "content": "HNSW indexing delivers sub-20ms latency.", "metadata": {"title": "A"}}]
            res = await assessor.assess(None, "what latency", chunks)
            self.assertEqual(res["quality"], "adequate")
            res_mock = await assessor.assess(self.mock_orch._get_llm_client(), "q", chunks)
            self.assertEqual(res_mock["quality"], "adequate")
        _run(run())

    def test_critic_offline_keeps_all(self):
        async def run():
            critic = ReflectionCritic()
            chunks = [{"id": f"c{i}", "content": f"chunk {i}", "metadata": {}} for i in range(4)]
            keep = await critic.critique(None, "q", chunks)
            self.assertEqual(keep, list(range(4)))
            keep_mock = await critic.critique(self.mock_orch._get_llm_client(), "q", chunks)
            self.assertEqual(keep_mock, list(range(4)))
        _run(run())

    def test_critic_and_assessor_with_empty_chunks(self):
        async def run():
            critic = ReflectionCritic()
            assessor = RetrievalAssessor()
            self.assertEqual(await critic.critique(None, "q", []), [])
            self.assertEqual((await assessor.assess(None, "q", []))["quality"], "adequate")
        _run(run())

    # ------------------------------------------------------------------
    # (d) Real-LLM agentic path
    # ------------------------------------------------------------------
    def test_router_real_llm_uses_llm_decision(self):
        async def run():
            orch = AgentOrchestrator(vector_store=self.vector_store, max_steps=5, llm_client=self.fake_llm)
            router = AdaptiveRouter(classify_fallback=orch.classify_intent)
            intent = await router.route("Explain architecture standards", user_role="standard_user", llm_client=self.fake_llm)
            self.assertEqual(intent["retrieval_mode"], "multi")
            self.assertIn("needs_retrieval", intent)
            self.assertIn("needs_hybrid", intent)
            self.assertIn("is_composite", intent)
        _run(run())

    def test_real_llm_agentic_execute(self):
        async def run():
            orch = AgentOrchestrator(vector_store=self.vector_store, max_steps=5, llm_client=self.fake_llm)
            resp = await orch.execute("Explain the PGVector HNSW architecture standards and safety policies.")
            # Post-guardrails still pass.
            self.assertTrue(resp.guardrail_metrics["pre_execution_passed"])
            self.assertTrue(resp.guardrail_metrics["is_grounded"])
            self.assertGreaterEqual(resp.guardrail_metrics["factual_grounding_score"], 0.20)
            # LLM-driven router + synthesis were actually invoked.
            self.assertGreater(self.fake_llm.structured_calls, 0)
            self.assertGreater(self.fake_llm.complete_calls, 0)
            # Retrieval executed (multi-variant expansion -> vector_search traces).
            self.assertIn("vector_search", [t.tool_name for t in resp.tool_traces])
            # Rerank / transform reasoning recorded.
            self.assertTrue(any("variant" in s.lower() or "re-rank" in s.lower() or "rerank" in s.lower() for s in resp.reasoning_steps))
            # Additive agentic metrics present; existing keys intact.
            self.assertEqual(resp.guardrail_metrics["retrieval_mode"], "multi")
            self.assertIn("retrieval_quality", resp.guardrail_metrics)
            self.assertGreaterEqual(resp.guardrail_metrics["query_variants"], 2)
            self.assertGreaterEqual(resp.guardrail_metrics["chunks_critiqued"], 1)
        _run(run())

    def test_real_llm_stream_emits_tokens(self):
        async def run():
            orch = AgentOrchestrator(vector_store=self.vector_store, max_steps=5, llm_client=self.fake_llm)
            events = [ev async for ev in orch.execute_stream("Explain the enterprise safety policy.")]
            kinds = [ev["event"] for ev in events]
            self.assertIn("reasoning_step", kinds)
            self.assertIn("tool_trace", kinds)
            self.assertIn("token", kinds)
            tokens = [ev["data"]["token"] for ev in events if ev["event"] == "token"]
            self.assertTrue(any(str(t).strip() for t in tokens))
            done = next(ev for ev in events if ev["event"] == "done")
            self.assertTrue(done["data"]["guardrail_metrics"]["is_grounded"])
            self.assertGreater(len(done["data"]["answer"]), 20)
        _run(run())


if __name__ == "__main__":
    unittest.main()