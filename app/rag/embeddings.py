import logging
import math
import random
import re
from typing import List

from app.config import settings

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class BaseEmbeddingProvider:
    """Interface for embedding providers."""

    #: Human-readable provider name.
    name = "base"

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into a list of vectors."""
        raise NotImplementedError


class DeterministicMockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic hash-based pseudo-random projection embeddings.

    Literally the historical fake-embedding algorithm: for the same input and
    dimension it always returns the exact same vector, so vector_store seeding
    and existing tests keep working unchanged.
    """

    name = "mock"

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        """
        Generates a deterministic, normalized semantic embedding vector.
        Uses token-based pseudo-random projection so that semantically similar texts
        (sharing keywords and sub-phrases) naturally yield high cosine similarity.
        """
        if not text:
            text = "__empty__"

        tokens = [w.lower() for w in re.findall(r"\w+", text) if len(w) > 1]
        if not tokens:
            tokens = [text.lower()]

        if HAS_NUMPY:
            accum = np.zeros(self.dimension, dtype=np.float64)
            for token in tokens:
                seed = abs(hash(token)) % (2**32)
                rng = np.random.RandomState(seed)
                w_vec = rng.randn(self.dimension).astype(np.float64)
                w_norm = np.linalg.norm(w_vec)
                if w_norm > 0:
                    accum += w_vec / w_norm

            # Add minor full-text entropy
            text_seed = abs(hash(text)) % (2**32)
            text_rng = np.random.RandomState(text_seed)
            t_vec = text_rng.randn(self.dimension).astype(np.float64)
            t_norm = np.linalg.norm(t_vec)
            if t_norm > 0:
                accum += 0.05 * (t_vec / t_norm)

            norm = float(np.linalg.norm(accum))
            if norm == 0.0:
                accum[0] = 1.0
                norm = 1.0
            return (accum / norm).tolist()
        else:
            accum = [0.0] * self.dimension
            for token in tokens:
                seed = abs(hash(token)) % (2**32)
                rnd = random.Random(seed)
                w_vec = [rnd.gauss(0, 1) for _ in range(self.dimension)]
                w_norm = math.sqrt(sum(x * x for x in w_vec)) or 1.0
                for i in range(self.dimension):
                    accum[i] += w_vec[i] / w_norm

            text_seed = abs(hash(text)) % (2**32)
            text_rnd = random.Random(text_seed)
            t_vec = [text_rnd.gauss(0, 1) for _ in range(self.dimension)]
            t_norm = math.sqrt(sum(x * x for x in t_vec)) or 1.0
            for i in range(self.dimension):
                accum[i] += 0.05 * (t_vec[i] / t_norm)

            norm = math.sqrt(sum(x * x for x in accum))
            if norm == 0.0:
                accum[0] = 1.0
                norm = 1.0
            return [float(x / norm) for x in accum]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI-compatible /v1/embeddings client (httpx).

    Uses ``settings.EMBEDDING_MODEL`` (default ``text-embedding-3-small``),
    ``settings.OPENAI_API_KEY`` and ``settings.OPENAI_BASE_URL``. When the
    configured dimension is smaller than the model's native output it is passed
    as the ``dimensions`` parameter for Matryoshka (MRL) truncation.
    """

    name = "openai"

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        dimension: int = 1536,
    ):
        self.model = model or settings.EMBEDDING_MODEL
        self.api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.base_url = (base_url or settings.OPENAI_BASE_URL).rstrip("/")
        self.dimension = dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        import httpx

        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": list(texts)}
        # MRL truncation: only request a smaller dimension when requested dim
        # differs from a known model native size (conservative).
        if self.dimension and self.dimension < 3072:
            payload["dimensions"] = self.dimension
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=settings.RETRIEVAL_TIMEOUT_SECONDS) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return [item["embedding"] for item in data["data"]]


class LocalSentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Optional local sentence-transformers provider.

    The import is guarded so the package can be absent. If it is not installed
    the provider falls back to the deterministic mock provider with a warning.
    """

    name = "local"

    def __init__(self, model: str = None, dimension: int = 1536):
        self.model = model or settings.EMBEDDING_MODEL
        self.dimension = dimension
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; local embedding provider "
                "falling back to deterministic mock provider. "
                "Install extras: pip install -r requirements-extras.txt"
            )
            self._model = "mock"
            return
        self._model = SentenceTransformer(self.model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        if self._model == "mock":
            return DeterministicMockEmbeddingProvider(self.dimension).embed(texts)
        import numpy as np

        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [np.asarray(v, dtype=np.float64).tolist() for v in vectors]


#: Registry mapping provider name -> class.
EMBEDDING_PROVIDERS = {
    "mock": DeterministicMockEmbeddingProvider,
    "openai": OpenAIEmbeddingProvider,
    "local": LocalSentenceTransformerProvider,
}


class EmbeddingService:
    def __init__(self, dimension: int = None, provider: str = None):
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        provider_name = provider or settings.EMBEDDING_PROVIDER
        self.provider_name = provider_name
        self.provider = self._build_provider(provider_name)
        # A mock provider is always kept so `_generate_vector` stays
        # deterministic and identical for vector_store seeding, regardless of
        # which runtime provider is selected.
        self._mock_provider = DeterministicMockEmbeddingProvider(self.dimension)

    def _build_provider(self, provider_name: str) -> BaseEmbeddingProvider:
        cls = EMBEDDING_PROVIDERS.get(provider_name)
        if cls is None:
            logger.warning(
                "Unknown EMBEDDING_PROVIDER=%r, falling back to 'mock'.",
                provider_name,
            )
            cls = DeterministicMockEmbeddingProvider
        if cls is OpenAIEmbeddingProvider:
            return cls(dimension=self.dimension)
        if cls is LocalSentenceTransformerProvider:
            return cls(dimension=self.dimension)
        return cls(dimension=self.dimension)

    def _generate_vector(self, text: str) -> List[float]:
        """
        Deterministic mock embedding (unchanged behavior).

        Delegates to the mock provider so vector_store seeding output is
        byte-for-byte identical to the previous hash-based implementation for
        the same input.
        """
        return self._mock_provider._embed_one(text)

    async def get_embedding(self, text: str) -> List[float]:
        """Generates a normalized unit vector embedding for input text."""
        return self.provider.embed([text])[0]

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized embeddings for a batch of texts."""
        if not texts:
            return []
        return self.provider.embed(texts)

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Computes cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        if HAS_NUMPY:
            a = np.asarray(vec_a, dtype=np.float64)
            b = np.asarray(vec_b, dtype=np.float64)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        else:
            dot = sum(x * y for x, y in zip(vec_a, vec_b))
            norm_a = math.sqrt(sum(x * x for x in vec_a))
            norm_b = math.sqrt(sum(y * y for y in vec_b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(dot / (norm_a * norm_b))