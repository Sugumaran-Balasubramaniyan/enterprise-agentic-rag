import unittest
import asyncio
from app.agent.orchestrator import AgentOrchestrator

class TestAgentOrchestrator(unittest.TestCase):
    def test_orchestrator_execution(self):
        async def run_test():
            orchestrator = AgentOrchestrator()
            res = await orchestrator.execute("How do we deploy deterministic guardrails?")
            self.assertIsNotNone(res.answer)
            self.assertTrue(len(res.reasoning_steps) > 0)
            self.assertTrue(res.guardrail_metrics["pre_execution_passed"])
        
        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()
