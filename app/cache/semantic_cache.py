"""
Semantic cache for exact/near-duplicate queries.

Queries are embedded and compared by cosine similarity against previously
cached entries; a hit is returned when similarity meets the configured
threshold (default 0.95). The in-process dict is guarded by an ``asyncio.Lock``.
An optional Redis mirror is used when ``REDIS_URL`` is set AND ``redis`` is
importable; any Redis error falls back to the in-process store. Methods never
raise: on any error ``get`` returns ``None`` and ``put`` is a no-op, while the
error is counted in ``self.errors``.

TTL: at most ``max_entries`` (default 512) entries or 1 hour; oldest/stale
entries are dropped first.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

from app.config import settings
from app.rag.embeddings import EmbeddingService

DEFAULT_REDIS_KEY = "rag:semantic_cache"


class SemanticCache:
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        max_entries: int = 512,
        ttl_seconds: int = 3600,
        threshold: Optional[float] = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.threshold = (
            threshold
            if threshold is not None
            else float(getattr(settings, "CACHE_SIMILARITY_THRESHOLD", 0.95))
        )

        self.hits: int = 0
        self.misses: int = 0
        self.errors: int = 0

        self._entries: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._redis = None
        self._use_redis = False
        self._init_redis()

    # -- Public API ---------------------------------------------------------

    async def get(self, query_text: str) -> Optional[dict]:
        """Return the cached response dict on a similarity hit, else None."""
        if not query_text:
            return None
        try:
            query_embedding = await self.embedding_service.get_embedding(query_text)
        except Exception:
            self.errors += 1
            return None

        entry = await self._find_best(query_embedding)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry["response"]

    async def put(
        self,
        query_text: str,
        payload: dict,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a cache entry for ``query_text``. Never raises."""
        if not query_text:
            return
        try:
            embedding = await self.embedding_service.get_embedding(query_text)
        except Exception:
            self.errors += 1
            return

        entry: Dict[str, Any] = {
            "query": query_text,
            "response": payload,
            "embedding": embedding,
            "ts": time.monotonic(),
            "metadata": metadata or {},
        }

        try:
            async with self._lock:
                self._entries[query_text] = entry
                self._trim(now=time.monotonic())
        except Exception:
            self.errors += 1
            return

        await self._mirror_to_redis(query_text, entry)

    # -- Internals ----------------------------------------------------------

    def _init_redis(self) -> None:
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return
        try:
            import redis.asyncio as redis_client

            self._redis = redis_client.from_url(redis_url)
            self._use_redis = True
        except Exception:
            self._redis = None
            self._use_redis = False

    async def _find_best(
        self, query_embedding: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Return the best matching in-proc entry at/above threshold."""
        try:
            async with self._lock:
                best: Optional[Dict[str, Any]] = None
                best_sim = 0.0
                now = time.monotonic()
                stale: List[str] = []
                for key, entry in self._entries.items():
                    if now - entry["ts"] > self.ttl_seconds:
                        stale.append(key)
                        continue
                    sim = EmbeddingService.cosine_similarity(
                        query_embedding, entry["embedding"]
                    )
                    if sim >= self.threshold and sim > best_sim:
                        best_sim = sim
                        best = entry
                for key in stale:
                    self._entries.pop(key, None)
                return best
        except Exception:
            self.errors += 1
            return None

    def _trim(self, now: float) -> None:
        # Drop stale entries (older than TTL).
        for key in [k for k, e in self._entries.items() if now - e["ts"] > self.ttl_seconds]:
            self._entries.pop(key, None)
        # Drop oldest entries beyond the max size.
        while len(self._entries) > self.max_entries:
            oldest = next(iter(self._entries))
            self._entries.pop(oldest, None)

    async def _mirror_to_redis(self, query_text: str, entry: Dict[str, Any]) -> None:
        if not self._use_redis or self._redis is None:
            return
        try:
            serialized = json.dumps(entry, default=str)
            await self._redis.hset(DEFAULT_REDIS_KEY, query_text, serialized)
        except Exception:
            self.errors += 1
            self._use_redis = False  # fall back to in-proc on any redis error