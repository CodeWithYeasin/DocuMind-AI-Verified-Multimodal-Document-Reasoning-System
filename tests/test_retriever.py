"""Unit tests for FAISS retriever."""

import numpy as np

from src.retriever.faiss_index import FaissRetriever
from src.utils.schemas import DocumentChunk


def test_faiss_retriever_returns_expected_top_hit() -> None:
    """Retriever should return the nearest chunk for a given query vector."""
    chunks = [
        DocumentChunk(chunk_id="c1", document_id="d1", text="alpha", metadata={}),
        DocumentChunk(chunk_id="c2", document_id="d1", text="beta", metadata={}),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    retriever = FaissRetriever(embedding_dim=2)
    retriever.add(chunks=chunks, embeddings=embeddings)

    query = np.array([[1.0, 0.0]], dtype=np.float32)
    results = retriever.search(query_embedding=query, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk"].chunk_id == "c1"
