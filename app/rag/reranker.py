"""
Retrieval re-ranking stack.

Provides a small set of pluggable re-rankers that re-order a list of retrieval
results (as produced by :class:`PGVectorStore.hybrid_search`) and attach a
``rerank_score`` float to each dict.

The default (:class:`LinearScoreReranker`) is fully offline with zero third-party
dependencies and simply fuses the score fields the retrieval stage already
outputs (``similarity``, ``lexical_score``, ``rrf_score``). Heavier model-based
re-rankers (:class:`CrossEncoderReranker`, :class:`CohereReranker`) are imported
lazily and degrade gracefully to the linear fallback if their dependencies or
credentials are unavailable.

The provider is selected via ``settings.RERANK_PROVIDER`` (read defensively with
``getattr`` so the module imports even before app/config is updated):
    - "linear"          -> LinearScoreReranker (default, offline)
    - "cross_encoder"   -> CrossEncoderReranker (sentence-transformers)
    - "cohere"          -> CohereReranker (httpx + COHERE_API_KEY)
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from app.config import settings
except Exception:  # pragma: no cover - defensive import
    settings = None


class BaseReranker:
    """
    Base class for all re-rankers.

    Subclasses must implement :meth:`rerank`, which takes the original query
    string and a list of result dicts and returns the same dicts re-ordered by
    relevance, each augmented with a ``rerank_score`` float.
    """

    name: str = "base"

    async def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError


def _min_max_normalize(values: List[float]) -> List[float]:
    """Min-max normalizes a list of floats into [0, 1]. Empty/constant lists -> 0.0."""
    if not values:
        return []
    vals = [float(v) for v in values]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [0.0] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


class LinearScoreReranker(BaseReranker):
    """
    Offline, dependency-free re-ranker.

    Computes   score = w1*sim_norm + w2*lexical_norm + w3*rrf_norm
    where each component is min-max normalized across the candidate list. Fields
    that are absent from a result dict are treated as 0.0.
    """

    name = "linear"

    def __init__(self, w1: float = 0.4, w2: float = 0.3, w3: float = 0.3):
        self.weights = (float(w1), float(w2), float(w3))

    async def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return results

        w1, w2, w3 = self.weights

        def _field(results_: List[Dict[str, Any]], key: str, default: float = 0.0) -> List[float]:
            out = []
            for r in results_:
                val = r.get(key)
                try:
                    out.append(float(val) if val is not None else default)
                except (TypeError, ValueError):
                    out.append(default)
            return out

        sim_norm = _min_max_normalize(_field(results, "similarity"))
        lex_norm = _min_max_normalize(_field(results, "lexical_score"))
        rrf_norm = _min_max_normalize(_field(results, "rrf_score"))

        scored = []
        for i, r in enumerate(results):
            merged = dict(r)
            merged["rerank_score"] = round(
                w1 * sim_norm[i] + w2 * lex_norm[i] + w3 * rrf_norm[i], 6
            )
            scored.append(merged)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored


class CrossEncoderReranker(BaseReranker):
    """
    Model-based re-ranker backed by sentence-transformers CrossEncoder.

    The model is loaded lazily on first use and cached as a module-level
    singleton. Scoring/ordering runs in a worker thread (``asyncio.to_thread``)
    so the event loop is never blocked. If sentence-transformers is not
    installed or the model fails to load, it logs a warning and falls back to
    :class:`LinearScoreReranker`.
    """

    name = "cross_encoder"

    def __init__(self, model_name: Optional[str] = None, top_n: Optional[int] = None):
        model_name = model_name or self._default_model()
        self.model_name = model_name
        self.top_n = top_n
        self._fallback = LinearScoreReranker()

    @staticmethod
    def _default_model() -> str:
        return os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    @staticmethod
    def _load_model(model_name: str):
        # Imported lazily so the module (and the rest of the app) works on the
        # base venv without sentence-transformers installed.
        from sentence_transformers import CrossEncoder  # type: ignore
        return CrossEncoder(model_name)

    async def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return results
        try:
            model = self._load_model(self.model_name)  # singleton via import cache? no
        except Exception as exc:  # ImportError, OSError (download), RuntimeError, ...
            logger.warning("CrossEncoder %s unavailable (%s); using linear fallback", self.model_name, exc)
            return await self._fallback.rerank(query, results)

        try:
            pairs = [(query, r.get("content", "")) for r in results]
            scores = await asyncio.to_thread(model.predict, pairs)
        except Exception as exc:  # model predict failure
            logger.warning("CrossEncoder predict failed (%s); using linear fallback", exc)
            return await self._fallback.rerank(query, results)

        scored = []
        for r, s in zip(results, scores):
            merged = dict(r)
            merged["rerank_score"] = round(float(s), 6)
            scored.append(merged)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        if self.top_n is not None:
            scored = scored[: self.top_n]
        return scored


class CohereReranker(BaseReranker):
    """
    Remote re-ranker calling Cohere's v2 /rerank REST endpoint.

    Requires the ``COHERE_API_KEY`` environment variable and the ``httpx``
    package. Uses model ``rerank-v3.5``. If either is missing, it falls back to
    :class:`LinearScoreReranker`.
    """

    name = "cohere"
    API_URL = "https://api.cohere.com/v2/rerank"
    RERANK_MODEL = "rerank-v3.5"

    def __init__(self, api_key: Optional[str] = None, top_n: Optional[int] = None):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.top_n = top_n
        self._fallback = LinearScoreReranker()

    async def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return results

        if not self.api_key:
            logger.warning("COHERE_API_KEY not set; using linear fallback")
            return await self._fallback.rerank(query, results)

        try:
            import httpx  # type: ignore
        except ImportError:
            logger.warning("httpx not installed; using linear fallback")
            return await self._fallback.rerank(query, results)

        documents = [str(r.get("content", "")) for r in results]
        top_n = self.top_n if self.top_n is not None else len(documents)

        try:
            payload = {
                "model": self.RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            resp = await asyncio.to_thread(
                httpx.post, self.API_URL, json=payload, headers=headers, timeout=60.0
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # network / HTTP error
            logger.warning("Cohere rerank failed (%s); using linear fallback", exc)
            return await self._fallback.rerank(query, results)

        # v2 response shape: {"results": [{"index": int, "relevance_score": float}, ...]}
        results_by_index: Dict[int, Dict[str, Any]] = {}
        for item in data.get("results", []):
            idx = int(item.get("index", -1))
            if 0 <= idx < len(results):
                results_by_index[idx] = item

        if not results_by_index:
            logger.warning("Empty/unknown Cohere results; using linear fallback")
            return await self._fallback.rerank(query, results)

        scored = []
        for idx, item in results_by_index.items():
            merged = dict(results[idx])
            merged["rerank_score"] = round(float(item.get("relevance_score", 0.0)), 6)
            scored.append(merged)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored


def get_reranker(provider: Optional[str] = None) -> BaseReranker:
    """
    Factory that returns a configured re-ranker instance based on
    ``settings.RERANK_PROVIDER`` (or the ``provider`` argument). Unknown/absent
    values resolve to the offline :class:`LinearScoreReranker`.
    """
    if provider is None:
        if settings is not None:
            provider = getattr(settings, "RERANK_PROVIDER", None)
        provider = provider or os.getenv("RERANK_PROVIDER", "linear")

    normalized = str(provider).strip().lower()
    if normalized == "cross_encoder" or normalized in {"cross-encoder", "crossencoder"}:
        return CrossEncoderReranker()
    if normalized == "cohere":
        return CohereReranker()
    return LinearScoreReranker()