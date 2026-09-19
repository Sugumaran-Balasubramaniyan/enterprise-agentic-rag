"""Pluggable LLM client layer.

Public API (the contract for the agent loop / orchestrator):

- ``ToolSpec``: declared function a model may call.
- ``LLMClient``: abstract async client exposing ``complete``, ``structured``,
  ``stream`` and ``chat_with_tools``.
- ``OpenAICompatibleClient``: httpx-based client for any OpenAI-compatible
  ``/chat/completions`` endpoint, with tenacity retries and cost tracking.
- ``MockLLMClient``: fully deterministic, network-free client.
- ``get_llm_client``: factory selecting a provider via ``LLM_PROVIDER``.
- Exceptions: ``LLMError``, ``LLMStructuredOutputError``, ``LLMToolCallError``.
- ``estimate_tokens``: cheap token estimator (tiktoken when available).
"""

from .client import (
    LLMClient,
    LLMError,
    LLMStructuredOutputError,
    LLMToolCallError,
    ToolSpec,
    estimate_tokens,
    get_llm_client,
)
from .mock import MockLLMClient
from .providers import OpenAICompatibleClient

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMStructuredOutputError",
    "LLMToolCallError",
    "ToolSpec",
    "estimate_tokens",
    "get_llm_client",
    "MockLLMClient",
    "OpenAICompatibleClient",
]