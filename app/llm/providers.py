"""OpenAI-compatible provider backed by httpx + tenacity retries.

Reads config defensively (``getattr`` + env fallback) because the settings
module is owned by a parallel agent and may not expose every attribute yet.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator, Optional

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_exponential

from .client import LLMClient

try:
    from app.config import settings
except Exception:  # pragma: no cover - config not importable in isolation
    settings = None

# Per-1K-token cost (USD) table. Extend as models are added.
PRICE_PER_1K_TOKENS: dict[str, float] = {
    "gpt-4.1-mini": 0.0004,
    "gpt-4o-mini": 0.00015,
    "deepseek-chat": 0.0002,
    "default": 0.001,
}


def _get_env(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val is not None and val != "" else default


def _setting(name: str, default: str) -> str:
    if settings is not None:
        val = getattr(settings, name, None)
        if val is not None and val != "":
            return str(val)
    return _get_env(name, default)


def _base_url() -> str:
    return _setting("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _model() -> str:
    return _setting("LLM_MODEL", "gpt-4.1-mini")


def _max_tokens() -> Optional[int]:
    try:
        return int(_setting("LLM_MAX_TOKENS", "4096"))
    except (TypeError, ValueError):
        return 4096


def _api_key() -> str:
    if settings is not None:
        val = getattr(settings, "LLM_API_KEY", None) or getattr(
            settings, "OPENAI_API_KEY", None
        )
        if val:
            return str(val)
    return _get_env("OPENAI_API_KEY", "")


def _parse_usage(data: dict, model: str) -> dict:
    usage = data.get("usage") or {}
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(usage.get("total_tokens", prompt + completion) or 0),
        "model": model,
        "provider": "openai",
    }


def _maybe_retry(exc: BaseException) -> bool:
    """Retry policy: transient HTTP errors (429, 5xx) and network failures."""
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429,) or exc.response.status_code >= 500
    return False


class OpenAICompatibleClient(LLMClient):
    """Async client for OpenAI-compatible ``/chat/completions`` endpoints.

    Parameters may come from the constructor (highest priority) or, failing
    that, from ``settings`` / environment variables — read defensively.

    Retries up to 3 times on transient HTTP/network errors with exponential
    backoff, honoring a ``Retry-After`` header when the server sends one.
    """

    provider: str = "openai"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
        default_headers: Optional[dict] = None,
        **_: object,
    ):
        self.base_url = (base_url or _base_url()).rstrip("/")
        self.api_key = api_key if api_key is not None else _api_key()
        self.model = model or _model()
        self.max_tokens = max_tokens
        if self.max_tokens is None:
            self.max_tokens = _max_tokens()
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        if default_headers:
            self.headers.update(default_headers)
        self._http = httpx.AsyncClient(timeout=self.timeout, headers=self.headers)
        self._last_usage: dict = {}
        self._retry_after_floor = 0.0

    async def aclose(self) -> None:
        await self._http.aclose()

    def estimate_cost(self, usage: dict, model: Optional[str] = None) -> float:
        """Estimate USD cost for a usage dict using :data:`PRICE_PER_1K_TOKENS`."""
        model = model or self.model
        price = PRICE_PER_1K_TOKENS.get(model, PRICE_PER_1K_TOKENS["default"])
        total = int(usage.get("total_tokens", 0) or 0)
        return round(total / 1000 * price, 6)

    # -- transport -----------------------------------------------------------

    def _build_payload(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        tools: Optional[list[dict]],
        tool_choice: Optional[str],
        response_format: Optional[dict],
        stream: bool,
    ) -> dict:
        payload: dict = {"model": self.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        token_cap = max_tokens if max_tokens is not None else self.max_tokens
        if token_cap:
            payload["max_tokens"] = token_cap
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if response_format is not None:
            payload["response_format"] = response_format
        if stream:
            payload["stream"] = True
        return payload

    def _wait(self, retry_state) -> float:
        """Exponential backoff, floored by any Retry-After the server sent."""
        base = wait_exponential(multiplier=0.5, max=8)(retry_state)
        return max(base, self._retry_after_floor)

    async def _post(self, payload: dict) -> httpx.Response:
        @retry(
            stop=stop_after_attempt(3),
            wait=self._wait,
            retry=retry_if_exception(_maybe_retry),
            reraise=True,
        )
        async def attempt() -> httpx.Response:
            try:
                resp = await self._http.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        self._retry_after_floor = float(retry_after)
                    except (TypeError, ValueError):
                        self._retry_after_floor = 1.0
                raise

        return await attempt()

    @staticmethod
    def _tool_calls_from_message(message: dict) -> Optional[list[dict]]:
        """Normalize OpenAI tool_calls into ``{id, name, arguments: dict}``."""
        raw = message.get("tool_calls")
        if not raw:
            return None
        calls: list[dict] = []
        for item in raw:
            fn = item.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = dict(args_raw or {})
            calls.append(
                {
                    "id": item.get("id") or f"call_{len(calls)}",
                    "name": fn.get("name", ""),
                    "arguments": args,
                }
            )
        return calls

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
        payload = self._build_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            stream=False,
        )
        resp = await self._post(payload)
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        tool_calls = self._tool_calls_from_message(message)
        self._last_usage = _parse_usage(data, self.model)
        return content, tool_calls, self._last_usage

    async def _stream_chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=None,
            tool_choice=None,
            response_format=response_format,
            stream=True,
        )
        prompt_tokens = 0
        completion_tokens = 0
        try:
            async with self._http.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens) or 0)
                        completion_tokens = int(
                            usage.get("completion_tokens", completion_tokens) or 0
                        )
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece
                        completion_tokens += max(1, len(piece) // 4)
        finally:
            self._last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "model": self.model,
                "provider": "openai",
            }