"""Adaptive-RAG routing and retrieval quality gates.

This module holds three small, well-typed components that let the agent loop
become genuinely LLM-driven when a real model is configured, while degrading to
exactly the deterministic behavior when only a mock client (or none) is
available:

- :class:`AdaptiveRouter`    - LLM-routed intent classification with a hard
                               deterministic fallback (:meth:`classify_intent`).
- :class:`RetrievalAssessor` - CRAG-style retrieval adequacy evaluator.
- :class:`ReflectionCritic`  - Self-RAG-lite relevance critique that selects
                               which retrieved chunks to keep.

Every LLM-facing method is wrapped in ``try/except`` and an ``asyncio.wait_for``
timeout so offline or hung providers can never block or crash the loop; on any
failure they return the deterministic fallback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Literal, Optional, Union

logger = logging.getLogger(__name__)

try:
    from app.config import settings
except Exception:  # pragma: no cover - defensive import
    settings = None

try:
    from pydantic import BaseModel, Field

    HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    HAS_PYDANTIC = False

    class BaseModel:  # type: ignore
        def __init__(self, **_):  # type: ignore
            pass

    Field = lambda default=None, **_: default  # noqa: E731

_MOCK_PROVIDERS = {"mock", "offline", "none", ""}


def is_mock_provider(llm_client: Any) -> bool:
    """True when ``llm_client`` is missing or backed by a mock/offline provider."""
    if llm_client is None:
        return True
    provider = getattr(llm_client, "provider", None)
    if provider is None:
        return True
    return str(provider).strip().lower() in _MOCK_PROVIDERS


def _timeout() -> float:
    if settings is not None:
        try:
            return float(getattr(settings, "RETRIEVAL_TIMEOUT_SECONDS", 30.0) or 30.0)
        except (TypeError, ValueError):
            pass
    return 30.0


async def _timed(awaitable, timeout: float):
    """Await ``awaitable`` under a hard timeout, raising asyncio.TimeoutError."""
    return await asyncio.wait_for(awaitable, timeout=timeout)


# ---------------------------------------------------------------------------
# Structured-output models
# ---------------------------------------------------------------------------
if HAS_PYDANTIC:

    class RouteDecision(BaseModel):
        """LLM output for intent routing."""

        needs_retrieval: bool = Field(True, description="Whether knowledge base retrieval is required.")
        needs_calculation: bool = Field(False, description="Whether a calculation is required.")
        needs_verification: bool = Field(True, description="Whether citation verification is required.")
        department: Optional[str] = Field(None, description="Target department filter, if known.")
        retrieval_mode: Literal["none", "single", "multi"] = Field(
            "single", description="Retrieval strategy: none / single query / multi-variant."
        )

    class RetrievalAssessment(BaseModel):
        """CRAG-style evaluation of whether retrieved context is adequate."""

        quality: Literal["adequate", "inadequate"] = Field("adequate", description="Retrieval adequacy verdict.")
        reason: str = Field("", description="Short justification for the verdict.")

    class ChunkCritique(BaseModel):
        """Self-RAG-lite relevance verdicts per retrieved chunk."""

        keep_indices: List[int] = Field(default_factory=list, description="Indices of chunks to keep.")

else:  # pragma: no cover - consumer without pydantic cannot route via LLM anyway

    class RouteDecision(BaseModel):  # type: ignore[no-redef]
        needs_retrieval = True
        needs_calculation = False
        needs_verification = True
        department = None
        retrieval_mode = "single"


# ---------------------------------------------------------------------------
# AdaptiveRouter
# ---------------------------------------------------------------------------
class AdaptiveRouter:
    """Routes a query to an intent dict, LLM-first with deterministic fallback.

    The deterministic path (``llm_client`` is ``None``/mock) returns the
    ``classify_fallback`` result *verbatim* so mock-mode behavior is byte-for-byte
    identical to the historical classifier.
    """

    def __init__(self, classify_fallback: Callable[..., dict]):
        self._classify = classify_fallback

    async def route(
        self,
        query: str,
        user_role: str = "standard_user",
        llm_client: Any = None,
    ) -> dict:
        """Return an intent dict (classify_intent shape) for ``query``."""
        if is_mock_provider(llm_client):
            return self._classify(query, user_role=user_role)

        try:
            decision = await self._route_with_llm(query, llm_client)
        except asyncio.TimeoutError:
            logger.warning("LLM router timed out; falling back to deterministic classifier")
            return self._classify(query, user_role=user_role)
        except Exception as exc:  # pragma: no cover - network/schema/provider
            logger.warning("LLM router failed (%s); falling back to deterministic classifier", exc)
            return self._classify(query, user_role=user_role)

        mode = decision.retrieval_mode or "single"
        return {
            "needs_retrieval": bool(decision.needs_retrieval),
            "needs_hybrid": mode == "multi",
            "needs_calculation": bool(decision.needs_calculation),
            "needs_verification": bool(decision.needs_verification),
            "department": decision.department,
            "is_composite": bool(decision.needs_retrieval and decision.needs_calculation),
            "retrieval_mode": mode,
        }

    async def _route_with_llm(self, query: str, llm_client: Any) -> RouteDecision:
        system = (
            "You are an enterprise RAG query router. Decide which agent "
            "capabilities a query needs: retrieval (knowledge base lookup), "
            "calculation (arithmetic / sizing), and verification (citation "
            "audit). retrieval_mode is 'none' when no lookup is needed, "
            "'single' for one retrieval query, or 'multi' when the query is "
            "ambiguous and benefits from multiple query variants. Set "
            "department to a known department name only when the query clearly "
            "references one, otherwise null. Reply only with the requested JSON."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
        decision, _meta = await _timed(
            llm_client.structured(messages=messages, response_model=RouteDecision),
            _timeout(),
        )
        return decision


# ---------------------------------------------------------------------------
# RetrievalAssessor (CRAG-style)
# ---------------------------------------------------------------------------
class RetrievalAssessor:
    """Evaluates whether retrieved context is adequate to answer the query."""

    async def assess(self, llm_client: Any, query: str, chunks: List[dict]) -> dict:
        """Return ``{'quality': 'adequate'|'inadequate', 'reason': str}``."""
        if is_mock_provider(llm_client) or not chunks:
            return {"quality": "adequate", "reason": "No real LLM or no chunks to assess."}

        try:
            verdict = await self._assess_with_llm(llm_client, query, chunks)
        except asyncio.TimeoutError:
            logger.warning("Retrieval assessor timed out; treating context as adequate")
            return {"quality": "adequate", "reason": "Assessor timeout; assumed adequate."}
        except Exception as exc:  # pragma: no cover
            logger.warning("Retrieval assessor failed (%s); assumed adequate", exc)
            return {"quality": "adequate", "reason": f"Assessor error: {exc}"}

        return {"quality": verdict.quality, "reason": verdict.reason or ""}

    async def _assess_with_llm(self, llm_client: Any, query: str, chunks: List[dict]) -> RetrievalAssessment:
        context = _render_chunks(chunks)
        system = (
            "You are a retrieval-quality assessor (CRAG). Review whether the "
            "retrieved document chunks contain enough relevant, non-contradictory "
            "information to fully answer the user's question. Reply 'adequate' if "
            "they do, or 'inadequate' if they are missing, sparse, or irrelevant. "
            "Keep reasons to one sentence. Reply only with the requested JSON."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Query:\n{query}\n\nRetrieved chunks:\n{context}"},
        ]
        verdict, _meta = await _timed(
            llm_client.structured(messages=messages, response_model=RetrievalAssessment),
            _timeout(),
        )
        return verdict


# ---------------------------------------------------------------------------
# ReflectionCritic (Self-RAG-lite)
# ---------------------------------------------------------------------------
class ReflectionCritic:
    """Drops irrelevant retrieved chunks via a single relevance call."""

    async def critique(self, llm_client: Any, query: str, chunks: List[dict]) -> List[int]:
        """Return the kept chunk indices (0-based); ALL kept when offline."""
        if is_mock_provider(llm_client) or not chunks:
            return list(range(len(chunks)))

        try:
            keep = await self._critique_with_llm(llm_client, query, chunks)
        except asyncio.TimeoutError:
            logger.warning("Reflection critique timed out; keeping all chunks")
            return list(range(len(chunks)))
        except Exception as exc:  # pragma: no cover
            logger.warning("Reflection critique failed (%s); keeping all chunks", exc)
            return list(range(len(chunks)))

        valid = list(range(len(chunks)))
        selected = []
        for i in keep:
            try:
                idx = int(i)
            except (TypeError, ValueError):
                continue
            if idx in valid and idx not in selected:
                selected.append(idx)

        # Safety: never return an empty working set on a real retrieval.
        return selected or list(range(len(chunks)))

    async def _critique_with_llm(self, llm_client: Any, query: str, chunks: List[dict]) -> List[int]:
        context = _render_chunks(chunks)
        system = (
            "You are a relevance critic. Given a question and a numbered list of "
            "retrieved document chunks, return the 0-based indices of the chunks "
            "that are directly relevant and supportive for answering the question. "
            "Return an empty list only if none are relevant. Reply only with the "
            "requested JSON."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question:\n{query}\n\nChunks:\n{context}"},
        ]
        critique, _meta = await _timed(
            llm_client.structured(messages=messages, response_model=ChunkCritique),
            _timeout(),
        )
        return [int(i) for i in (critique.keep_indices or [])]


def _render_chunks(chunks: Union[List[dict], List[str]]) -> str:
    """Render chunks as a numbered, title-tagged block for LLM consumption."""
    lines = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, str):
            content, title = chunk, ""
        else:
            meta = chunk.get("metadata") or {}
            title = meta.get("title") or meta.get("source") or chunk.get("document_id", "")
            content = chunk.get("content", "")
        tag = f"[{title}] " if title else ""
        lines.append(f"{i}. {tag}{content}")
    return "\n".join(lines)