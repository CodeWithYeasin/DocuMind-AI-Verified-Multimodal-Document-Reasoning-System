"""Sentence embedding service for retrieval."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)


class EmbeddingService:
    """Create dense embeddings for text chunks and queries."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Load sentence-transformer model."""
        LOGGER.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        return int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Generate normalized embeddings for multiple texts."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate normalized embedding for a query."""
        if not query.strip():
            raise ValueError("query must be non-empty")
        vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)
