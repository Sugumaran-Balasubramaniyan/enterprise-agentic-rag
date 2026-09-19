"""Core LLM client abstraction: tool specs, base client, exceptions, factory.

The base :class:`LLMClient` implements the orchestration logic shared by every
provider — structured-output retry loop and the tool-calling loop — while
deferring only the raw transport to concrete subclasses via ``_chat`` and
``_stream_chat``.
"""

from __future__ import annotations

import abc
import json
from typing import Any, AsyncIterator, Callable, Optional

from pydantic import BaseModel, Field, ValidationError

try:
    from app.config import settings
except Exception:  # pragma: no cover - config not importable in isolation
    settings = None


class LLMError(Exception):
    """Base error for the LLM layer."""


class LLMStructuredOutputError(LLMError):
    """Raised when structured output cannot be parsed/validated after retries."""


class LLMToolCallError(LLMError):
    """Raised when a tool call in the agent loop fails irrecoverably."""


class ToolSpec(BaseModel):
    """Declarative description of a callable tool.

    ``parameters`` is a JSON-Schema object describing the function arguments
    (e.g. ``{"type": "object", "properties": {...}, "required": [...]}``). If
    omitted, an empty object schema is substituted.
    """

    name: str
    description: str = ""
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})

    def to_openai(self) -> dict:
        """Return the OpenAI ``tools=[...]`` element for this spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def estimate_tokens(text: str) -> int:
    """Estimate token count, preferring tiktoken (cl100k_base) when available.

    Falls back to a ``len(text) // 4`` heuristic so the layer works offline.
    """
    if not isinstance(text, str):
        text = str(text)
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # pragma: no cover - tiktoken optional
        return max(1, len(text) // 4)


def _strip_code_fences(text: str) -> str:
    """Strip ```json ... ``` wrappers a model may add around its output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


class LLMClient(abc.ABC):
    """Abstract pluggable LLM client.

    Concrete subclasses must implement:

    - ``_chat(messages, *, temperature, max_tokens, tools, tool_choice,
      response_format)`` returning ``(content, tool_calls, usage)`` where
      ``content`` is ``str | None``, ``tool_calls`` is a list of
      ``{"id": str, "name": str, "arguments": dict}`` or ``None``, and ``usage``
      is a dict with ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``.
    - ``_stream_chat(messages, ...)`` an async iterator of content chunks.

    All orchestration (structured retries, tool loop) lives here.
    """

    provider: str = "base"
    model: str = ""

    # -- transport hooks (implemented by subclasses) -------------------------

    @abc.abstractmethod
    async def _chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> tuple[Optional[str], Optional[list[dict]], dict]:
        """One non-streaming model turn. Return (content, tool_calls, usage)."""

    @abc.abstractmethod
    def _stream_chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        """Stream content chunks for the given messages."""

    # -- public API ----------------------------------------------------------

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """One non-tool completion. Returns the canonical result dict."""
        content, _tool_calls, usage = await self._chat(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        return self._result(content or "", usage)

    async def structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[BaseModel, dict]:
        """Parse a model response into ``response_model``.

        Retries up to 3 times on JSON-decoding / validation errors, feeding the
        previous bad output and the error back into the conversation so the
        model can fix its JSON. Raises :class:`LLMStructuredOutputError` if all
        attempts fail.
        """
        failures: list[tuple[str, Exception]] = []
        working = [dict(m) for m in messages]

        for attempt in range(3):
            if attempt > 0:
                bad_content, err = failures[-1]
                working = [dict(m) for m in messages] + [
                    {"role": "assistant", "content": bad_content or ""},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be interpreted. "
                            "Return ONLY valid JSON matching the requested "
                            "schema, with no surrounding text or code fences.\n"
                            f"Previous output: {bad_content!r}\n"
                            f"Error: {err}"
                        ),
                    },
                ]

            content, _tc, usage = await self._chat(
                working,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = content or ""
            cleaned = _strip_code_fences(content)

            try:
                obj = json.loads(cleaned)
            except (json.JSONDecodeError, TypeError) as exc:
                failures.append((content, exc))
                continue

            try:
                parsed = response_model.model_validate(obj)
            except ValidationError as exc:
                failures.append((content, exc))
                continue

            meta = self._result(content, usage)
            meta["validation"] = {"attempts": attempt + 1, "ok": True}
            return parsed, meta

        _last_content, last_err = failures[-1]
        raise LLMStructuredOutputError(
            "Structured output failed after 3 attempts: "
            f"{type(last_err).__name__}: {last_err}"
        )

    async def stream(
        self,
        messages: list[dict],
        *,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Yield content chunks; records usage if the provider reports any."""
        async for chunk in self._stream_chat(messages, max_tokens=max_tokens):
            yield chunk

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolSpec],
        *,
        execute_tool: Callable[[str, dict], Any],
        max_iterations: int = 8,
        temperature: Optional[float] = None,
    ) -> dict:
        """Run the agent tool loop.

        Sends ``messages`` plus ``tools``; whenever the model answers with
        ``tool_calls``, each is executed through ``execute_tool(name, args)``
        (an async resolver returning a result string) and the results are
        appended using the OpenAI ``role="tool"`` protocol, then the loop
        repeats until the model returns final content or ``max_iterations`` is
        reached.
        """
        if execute_tool is None:

            async def _noop(_name: str, _args: dict) -> str:
                raise LLMToolCallError("No tool resolver was provided.")

            execute_tool = _noop

        openai_tools = [t.to_openai() for t in tools]
        transcript: list[dict] = [dict(m) for m in messages]
        total_tool_calls = 0
        usage: dict = {}

        for _ in range(max_iterations):
            content, tool_calls, usage = await self._chat(
                transcript,
                temperature=temperature,
                tools=openai_tools,
                tool_choice="auto",
            )

            if tool_calls:
                assistant_msg: dict = {"role": "assistant", "content": content or ""}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ]
                transcript.append(assistant_msg)
                total_tool_calls += len(tool_calls)

                for tc in tool_calls:
                    result = await execute_tool(tc["name"], tc["arguments"])
                    transcript.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": str(result) if result is not None else "",
                        }
                    )
                continue  # loop again for the model's reaction to tool results

            # No tool calls -> final content.
            transcript.append({"role": "assistant", "content": content or ""})
            return {
                "content": content or "",
                "messages": transcript,
                "tool_calls": total_tool_calls,
                "usage": usage or {},
                "final": True,
            }

        # Hit max_iterations without a final answer.
        return {
            "content": transcript[-1].get("content", "") if transcript else "",
            "messages": transcript,
            "tool_calls": total_tool_calls,
            "usage": usage or {},
            "final": False,
        }

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _result(content: str, usage: dict) -> dict:
        usage = usage or {}
        return {
            "content": content,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            },
            "model": usage.get("model", "") or "",
            "provider": usage.get("provider", "") or "",
        }

    async def aclose(self) -> None:  # optional override
        pass


def get_llm_client(**kwargs) -> LLMClient:
    """Factory: build a client from ``LLM_PROVIDER`` (+ optional overrides).

    Provider is read defensively — never assume the setting exists at import.
    """
    from .mock import MockLLMClient
    from .providers import OpenAICompatibleClient

    provider = "mock"
    if settings is not None:
        provider = str(getattr(settings, "LLM_PROVIDER", "mock") or "mock").lower()
    provider = str(kwargs.pop("provider", provider)).lower()

    if provider in ("openai", "openai-compatible", "openai_compatible"):
        return OpenAICompatibleClient(**kwargs)
    return MockLLMClient(**kwargs)