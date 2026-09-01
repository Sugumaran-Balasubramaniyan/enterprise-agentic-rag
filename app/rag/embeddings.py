import math
import random
import re
from typing import List
from app.config import settings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class EmbeddingService:
    def __init__(self, dimension: int = None, provider: str = None):
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.provider = provider or settings.LLM_PROVIDER

    def _generate_vector(self, text: str) -> List[float]:
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

    async def get_embedding(self, text: str) -> List[float]:
        """Generates a normalized unit vector embedding for input text."""
        return self._generate_vector(text)

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized embeddings for a batch of texts."""
        return [self._generate_vector(t) for t in texts]

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
