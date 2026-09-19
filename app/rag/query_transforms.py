"""
Query transformation utilities (query rewrite, expansion, HyDE).

Each transform is an optional, LLM-powered step that runs *before* retrieval to
produce query variants that surface better results from both the dense and sparse
branches of hybrid search. They interoperate with any client exposing an async
``.structured(messages, response_model)`` method (the app's LLMClient).

All transforms are strictly additive and safe offline: if ``llm_client`` is
``None``, or the configured provider is a mock, the transform returns input
unchanged. This keeps every function usable (and unit-testable) with zero LLM
dependency.

The :class:`LLMClient` type is imported lazily inside a try/except so this module
imports cleanly even before ``app.llm`` exists in the codebase.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from app.config import settings
except Exception:  # pragma: no cover - defensive import
    settings = None

# Lazy/optional import: never required for the module to load.
try:
    from app.llm.client import LLMClient  # type: ignore
except Exception:  # pragma: no cover - app.llm may not exist yet
    LLMClient = None  # type: ignore

try:
    from pydantic import BaseModel as _BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    HAS_PYDANTIC = False

    class _BaseModel:  # type: ignore
        """Minimal stand-in so the module imports without pydantic installed."""
        def __init__(self, **_):  # type: ignore
            pass

    Field = lambda default=None, **_: default  # noqa: E731 - stub for offline mode


if HAS_PYDANTIC:

    class RewriteOutput(_BaseModel):  # type: ignore[no-redef]
        """Structured output of the rewrite transform."""
        rewritten_query: Optional[str] = Field(
            None, description="A retrieval-optimized version of the original query."
        )

    class ExpandOutput(_BaseModel):  # type: ignore[no-redef]
        """Structured output of the expand transform."""
        queries: Optional[List[str]] = Field(
            None, description="List of paraphrases/variants of the original query."
        )

    class HydeOutput(_BaseModel):  # type: ignore[no-redef]
        """Structured output of the hypothetical-document (HyDE) transform."""
        document: Optional[str] = Field(
            None, description="A hypothetical answer/document matching the query."
        )
else:  # pragma: no cover

    class RewriteOutput(_BaseModel):  # type: ignore[no-redef]
        rewritten_query = None

    class ExpandOutput(_BaseModel):  # type: ignore[no-redef]
        queries = None

    class HydeOutput(_BaseModel):  # type: ignore[no-redef]
        document = None


def _is_offline(llm_client) -> bool:
    """True when the client is missing or the provider is a mock."""
    if llm_client is None:
        return True
    provider = getattr(llm_client, "provider", None) or getattr(
        settings, "LLM_PROVIDER", None
    )
    return str(provider).strip().lower() in {"mock", "offline", "none"}


class QueryTransformer:
    """
    Applies LLM-powered query transforms for retrieval.

    Parameters
    ----------
    enabled : dict[str, bool], optional
        Per-transform on/off switches. Keys: ``rewrite``, ``expand``, ``hyde``.
        Defaults to all enabled.
    expand_n : int, optional
        Number of paraphrases requested by :meth:`expand`.
    """

    def __init__(
        self,
        enabled: Optional[dict] = None,
        expand_n: int = 3,
    ):
        self.enabled = enabled or {"rewrite": True, "expand": True, "hyde": True}
        self.expand_n = max(1, int(expand_n))

    # ------------------------------------------------------------------ utils
    @staticmethod
    async def _structured(llm_client, response_model, messages) -> Optional[object]:
        """Best-effort structured LLM call; returns None on any failure."""
        if llm_client is None or not hasattr(llm_client, "structured"):
            return None
        try:
            result = await llm_client.structured(messages=messages, response_model=response_model)
            # Providers return (model, meta); clients may return the model alone.
            if isinstance(result, tuple) and result:
                result = result[0]
            return result
        except Exception as exc:  # network, schema, provider errors
            logger.warning("Structured LLM call failed (%s); using input unchanged", exc)
            return None

    @staticmethod
    def _unwrap(result, field: str, default):
        """Extracts ``field`` from a pydantic/object result, tolerating absence."""
        if result is None:
            return default
        value = getattr(result, field, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        if isinstance(value, (list, tuple)) and not value:
            return default
        return value

    # ------------------------------------------------------------- transforms
    async def rewrite(self, llm_client, query: str) -> str:
        """
        Produces a single retrieval-optimized version of ``query``.

        Falls back to returning ``query`` unchanged when offline or on error.
        """
        if not query or not query.strip():
            return query
        if not self.enabled.get("rewrite", True) or _is_offline(llm_client):
            return query
        if not HAS_PYDANTIC:
            return query

        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's question into a clear, self-contained, "
                    "keyword-rich retrieval query that maximizes both embedding "
                    "similarity and lexical keyword recall. Keep the original intent "
                    "and entity names. Reply only with the rewritten query."
                ),
            },
            {"role": "user", "content": query},
        ]
        result = await self._structured(llm_client, RewriteOutput, messages)
        return self._unwrap(result, "rewritten_query", query)

    async def expand(self, llm_client, query: str, n: int = 3) -> List[str]:
        """
        Produces up to ``n`` paraphrases/variants of ``query``.

        Offline/error behaviour: returns ``[query]`` only (no expansion).
        """
        q = query or ""
        if not q.strip():
            return [q]
        if not self.enabled.get("expand", True) or _is_offline(llm_client):
            return [q]
        if not HAS_PYDANTIC:
            return [q]

        n = max(1, int(n)) if n else self.expand_n

        messages = [
            {
                "role": "system",
                "content": (
                    f"Generate exactly {n} distinct paraphrases of the user's question. "
                    "Each paraphrase must keep the original meaning but use different "
                    "words (synonyms, restructured phrasing, broader/specific terms). "
                    "Return them as a JSON list of strings. Do not include numbering or "
                    "explanations."
                ),
            },
            {"role": "user", "content": q},
        ]
        result = await self._structured(llm_client, ExpandOutput, messages)
        queries = self._unwrap(result, "queries", [q])
        if not isinstance(queries, list):
            return [q]
        cleaned = [str(x).strip() for x in queries if str(x).strip()]
        if not cleaned:
            return [q]
        # Guarantee the original is always present, then cap at n.
        out = [q] if q not in cleaned else []
        out.extend(cleaned)
        return out[:n]

    async def hyde(self, llm_client, query: str) -> str:
        """
        Generates a hypothetical document (answer) for ``query`` (HyDE).

        Fallback/offline: returns ``query`` unchanged.
        """
        if not query or not query.strip():
            return query
        if not self.enabled.get("hyde", True) or _is_offline(llm_client):
            return query
        if not HAS_PYDANTIC:
            return query

        messages = [
            {
                "role": "system",
                "content": (
                    "Write a concise hypothetical document that would fully answer the "
                    "user's question, as if retrieved from a knowledge base. Use factual, "
                    "neutral, encyclopedic prose with concrete domain keywords. 2-4 "
                    "sentences."
                ),
            },
            {"role": "user", "content": query},
        ]
        result = await self._structured(llm_client, HydeOutput, messages)
        return self._unwrap(result, "document", query)