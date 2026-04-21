"""Main orchestration pipeline for document reasoning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.llm import LLMService
from src.parser.image_parser import ImageParser
from src.parser.pdf_parser import PDFParser
from src.retriever.embedding import EmbeddingService
from src.retriever.faiss_index import FaissRetriever
from src.utils.schemas import DocumentChunk
from src.verifier.checker import AnswerVerifier

LOGGER = logging.getLogger(__name__)


class DocumentReasoningPipeline:
    """End-to-end pipeline: parse -> retrieve -> answer -> verify."""

    def __init__(self, data_dir: str) -> None:
        """Initialize all pipeline modules."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.pdf_parser = PDFParser()
        self.image_parser = ImageParser()
        self.embedding_service = EmbeddingService()
        self.retriever = FaissRetriever(embedding_dim=self.embedding_service.dimension)
        self.llm_service = LLMService()
        self.verifier = AnswerVerifier()

    def ingest_document(self, file_path: str, document_id: str) -> int:
        """Ingest one document and index its chunks.

        Args:
            file_path: Absolute file path.
            document_id: Document identifier.

        Returns:
            Number of indexed chunks.
        """
        extension = Path(file_path).suffix.lower()
        if extension == ".pdf":
            chunks = self.pdf_parser.parse(file_path=file_path, document_id=document_id)
        elif extension in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            chunks = self.image_parser.parse(file_path=file_path, document_id=document_id)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        if not chunks:
            raise ValueError("No text could be extracted from the document")

        embeddings = self.embedding_service.embed_texts([chunk.text for chunk in chunks])
        self.retriever.add(chunks=chunks, embeddings=embeddings)
        LOGGER.info("Ingested document %s with %d chunks", document_id, len(chunks))
        return len(chunks)

    def ask(self, question: str, top_k: int = 5, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer a question using retrieved evidence and verification."""
        if not question.strip():
            raise ValueError("question must be non-empty")

        query_embedding = self.embedding_service.embed_query(question)
        hits = self.retriever.search(query_embedding=query_embedding, top_k=top_k, document_id=document_id)
        evidence_chunks: List[DocumentChunk] = [hit["chunk"] for hit in hits]

        answer = self.llm_service.generate_answer(question=question, evidence_chunks=evidence_chunks)
        verification = self.verifier.verify(answer=answer, evidence_chunks=evidence_chunks, question=question)

        evidence = [
            {
                "chunk_id": hit["chunk"].chunk_id,
                "document_id": hit["chunk"].document_id,
                "text": hit["chunk"].text,
                "score": hit["score"],
                "metadata": hit["chunk"].metadata,
            }
            for hit in hits
        ]

        return {
            "answer": answer,
            "evidence": evidence,
            "confidence_score": verification.confidence_score,
            "verification_label": verification.label,
            "verification": verification.model_dump(),
        }
