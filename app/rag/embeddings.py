import math
import random
from typing import List
from app.config import settings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class EmbeddingService:
    def __init__(self):
        self.dimension = settings.EMBEDDING_DIMENSION
        self.provider = settings.LLM_PROVIDER

    async def get_embedding(self, text: str) -> List[float]:
        if HAS_NUMPY:
            rng = np.random.RandomState(abs(hash(text)) % (2**32))
            vector = rng.randn(self.dimension)
            norm = np.linalg.norm(vector)
            return (vector / norm).tolist()
        else:
            rnd = random.Random(abs(hash(text)) % (2**32))
            raw = [rnd.gauss(0, 1) for _ in range(self.dimension)]
            norm = math.sqrt(sum(x*x for x in raw)) or 1.0
            return [x / norm for x in raw]

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.get_embedding(t) for t in texts]
