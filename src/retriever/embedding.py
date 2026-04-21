"""Embedding model abstraction for semantic retrieval."""

from __future__ import annotations

import hashlib
import logging
from typing import Iterable, List

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Generate embeddings with sentence-transformers and a deterministic fallback."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_sentence_transformers: bool = True,
        vector_dim: int = 384,
    ) -> None:
        """Initialize embedding model backend."""
        self.model_name = model_name
        self.vector_dim = vector_dim
        self._model = None

        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                self.vector_dim = int(self._model.get_sentence_embedding_dimension())
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("SentenceTransformer unavailable, using hash embeddings: %s", exc)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Encode input texts into normalized float32 vectors."""
        items: List[str] = [text if isinstance(text, str) else "" for text in texts]
        if not items:
            return np.zeros((0, self.vector_dim), dtype=np.float32)

        if self._model is not None:
            vectors = self._model.encode(
                items,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.astype(np.float32)

        vectors = np.vstack([self._hash_embedding(text) for text in items]).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        return vectors / norms

    def _hash_embedding(self, text: str) -> np.ndarray:
        """Create deterministic embedding from repeated SHA-256 digests."""
        data = text.encode("utf-8")
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        offset = 0

        while offset < self.vector_dim:
            digest = hashlib.sha256(data).digest()
            for byte in digest:
                if offset >= self.vector_dim:
                    break
                vec[offset] = (byte / 127.5) - 1.0
                offset += 1
            data = digest
        return vec
