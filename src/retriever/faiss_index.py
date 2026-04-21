"""FAISS index wrapper for chunk retrieval."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from src.utils.schemas import DocumentChunk

LOGGER = logging.getLogger(__name__)


class FaissRetriever:
    """FAISS-based vector retriever for document chunks."""

    def __init__(self, embedding_dim: int) -> None:
        """Initialize retriever with cosine similarity via inner product."""
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[DocumentChunk] = []

    @property
    def count(self) -> int:
        """Return number of indexed chunks."""
        return self.index.ntotal

    def add(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add chunks and corresponding embeddings to FAISS index."""
        if len(chunks) == 0:
            return
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings count must match chunks count")

        vectors = np.asarray(embeddings, dtype=np.float32)
        self.index.add(vectors)
        self.chunks.extend(chunks)
        LOGGER.info("Added %d chunks to index (total=%d)", len(chunks), self.count)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search top-k relevant chunks, with optional document filtering."""
        if self.count == 0:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("query_embedding must have shape (1, d)")

        search_k = min(max(top_k * 5, top_k), self.count)
        scores, indices = self.index.search(query, search_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if document_id and chunk.document_id != document_id:
                continue
            results.append({"chunk": chunk, "score": float(score)})
            if len(results) >= top_k:
                break

        return results

    def save(self, index_path: str, chunks_path: str) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(chunks_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(chunks_path, "w", encoding="utf-8") as handle:
            json.dump([chunk.to_dict() for chunk in self.chunks], handle, ensure_ascii=False, indent=2)

    def load(self, index_path: str, chunks_path: str) -> None:
        """Load FAISS index and chunk metadata from disk."""
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.chunks = [DocumentChunk.from_dict(entry) for entry in payload]
