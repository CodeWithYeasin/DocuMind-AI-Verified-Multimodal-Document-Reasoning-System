"""FAISS-backed semantic retrieval with numpy fallback."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np

from src.retriever.embedding import EmbeddingModel
from src.utils.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """Manage vector index and top-k retrieval for document chunks."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        """Initialize retriever and optional FAISS index backend."""
        self.embedding_model = embedding_model
        self.chunks: List[DocumentChunk] = []
        self.embeddings = np.zeros((0, self.embedding_model.vector_dim), dtype=np.float32)
        self.index = None
        self._faiss_available = False

        try:
            import faiss  # type: ignore

            self.index = faiss.IndexFlatIP(self.embedding_model.vector_dim)
            self._faiss_available = True
        except (ImportError, OSError) as exc:
            logger.warning("FAISS unavailable, using numpy retrieval: %s", exc)

    def build(self, chunks: Sequence[DocumentChunk]) -> None:
        """Build index from parsed chunks."""
        self.chunks = list(chunks)
        texts = [chunk.text for chunk in self.chunks]
        self.embeddings = self.embedding_model.encode(texts)

        if self._faiss_available and self.index is not None:
            self.index.reset()
            if len(self.embeddings) > 0:
                self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve top-k matching chunks with similarity scores."""
        if not self.chunks:
            return []

        query_vec = self.embedding_model.encode([query])
        top_k = min(top_k, len(self.chunks))

        if self._faiss_available and self.index is not None:
            scores, indices = self.index.search(query_vec, top_k)
            return [
                (self.chunks[idx], float(score))
                for idx, score in zip(indices[0].tolist(), scores[0].tolist())
                if idx >= 0
            ]

        sims = np.dot(self.embeddings, query_vec[0])
        ranked = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[idx], float(sims[idx])) for idx in ranked.tolist()]
