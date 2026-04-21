"""Unit tests for embedding and retrieval components."""

from __future__ import annotations

import unittest

from src.retriever.embedding import EmbeddingModel
from src.retriever.faiss_index import FAISSRetriever
from src.utils.schemas import DocumentChunk


class RetrieverTests(unittest.TestCase):
    """Validate retrieval ranking behavior."""

    def test_retriever_returns_relevant_chunk(self) -> None:
        """Retriever should rank semantically related chunk first."""
        model = EmbeddingModel(use_sentence_transformers=False, vector_dim=64)
        retriever = FAISSRetriever(embedding_model=model)

        chunks = [
            DocumentChunk(id="1", source="x", text="The capital of France is Paris."),
            DocumentChunk(id="2", source="x", text="Bananas are yellow fruits."),
            DocumentChunk(id="3", source="x", text="Python is a programming language."),
        ]
        retriever.build(chunks)
        top = retriever.retrieve("What is the capital of France?", top_k=1)

        self.assertEqual(len(top), 1)
        self.assertEqual(top[0][0].id, "1")


if __name__ == "__main__":
    unittest.main()
