"""Deterministic, network-free mock LLM client.

Used when ``LLM_PROVIDER`` is ``"mock"`` (the default) or for offline tests.
Everything is canned and repeatable — no randomness, no I/O.
"""

from __future__ import annotations

import re
from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel

from .client import LLMClient, ToolSpec

_BASE64_RE = re.compile(r"[A-Za-z0-9+/=\s]{20,}")

# Fixed answer when the last user message is "base64-like" nonsense.
_BASE64_RESPONSE = (
    "Mock answer: decoded placeholder content — no meaningful text was detected."
)


def _fabricate_value(schema: Optional[dict], *, depth: int = 0) -> Any:
    """Deterministically fabricate a value from a JSON-Schema property."""
    schema = schema or {}
    if depth > 3:
        return None
    stype = schema.get("type")

    if stype == "string":
        enum = schema.get("enum")
        if enum:
            return enum[0]
        default = schema.get("default")
        return default if default is not None else "mock_value"
    if stype == "integer" or stype == "number":
        default = schema.get("default")
        return default if default is not None else (0.0 if stype == "number" else 1)
    if stype == "boolean":
        default = schema.get("default")
        return default if default is not None else False
    if stype == "array":
        return []
    if stype == "object" or schema.get("properties") is not None:
        props = schema.get("properties") or {}
        return {
            key: _fabricate_value(prop, depth=depth + 1)
            for key, prop in props.items()
        }
    if stype is None:
        # $ref / anyOf / allOf — unable to resolve without the model, be safe.
        return None
    return None


class MockLLMClient(LLMClient):
    """Fully deterministic mock implementing the full LLMClient contract."""

    provider: str = "mock"
    model: str = "mock-model"

    def _usage(self, prompt: int = 10, completion: int = 10) -> dict:
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "model": self.model,
            "provider": self.provider,
        }

    @staticmethod
    def _last_user_content(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts = [
                        str(p.get("text", "")) if isinstance(p, dict) else str(p)
                        for p in content
                    ]
                    content = " ".join(parts)
                return str(content)
        return ""

    def _answer(self, messages: list[dict]) -> str:
        text = self._last_user_content(messages)
        if _BASE64_RE.search(text):
            return _BASE64_RESPONSE
        return f"Mock answer: {text}" if text else "Mock answer"

    # -- transport -----------------------------------------------------------

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
        if tools:
            first = tools[0]
            fn = first.get("function") or {}
            params = fn.get("parameters") or {}
            props = params.get("properties") or {}
            args = {
                name: _fabricate_value(spec)
                for name, spec in props.items()
            }
            calls = [{"id": "mock_call_0", "name": fn.get("name", ""), "arguments": args}]
            return None, calls, self._usage()
        return self._answer(messages), None, self._usage()

    async def _stream_chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        content = self._answer(messages)
        for word in content.split(" "):
            yield word + " "
        self._last_usage = self._usage()
        yield ""

    # -- helpers -------------------------------------------------------------

    async def structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[BaseModel, dict]:
        """Fabricate the model from its JSON schema (deterministic)."""
        schema = response_model.model_json_schema()
        props = schema.get("properties") or {}
        payload = {name: _fabricate_value(spec) for name, spec in props.items()}
        parsed = response_model.model_validate(payload)
        meta = self._result("", self._usage())
        meta["validation"] = {"method": "deterministic-fabrication", "ok": True}
        return parsed, meta

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolSpec],
        *,
        execute_tool: Any,
        max_iterations: int = 8,
        temperature: Optional[float] = None,
    ) -> dict:
        """Simulate ONE tool call, then return final content."""
        if not tools:
            content = self._answer(messages)
            return {
                "content": content,
                "messages": [*messages, {"role": "assistant", "content": content}],
                "tool_calls": 0,
                "usage": self._usage(),
                "final": True,
            }

        spec = tools[0]
        params = spec.parameters or {}
        props = params.get("properties") or {}
        args = {name: _fabricate_value(value) for name, value in props.items()}

        call_id = "mock_call_0"
        assistant_msg: dict = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "arguments": __import__("json").dumps(args),
                    },
                }
            ],
        }
        result = await execute_tool(spec.name, args)
        transcript = [
            *[dict(m) for m in messages],
            assistant_msg,
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(result) if result is not None else "",
            },
            {"role": "assistant", "content": "Mock answer"},
        ]
        return {
            "content": "Mock answer",
            "messages": transcript,
            "tool_calls": 1,
            "usage": self._usage(),
            "final": True,
        }