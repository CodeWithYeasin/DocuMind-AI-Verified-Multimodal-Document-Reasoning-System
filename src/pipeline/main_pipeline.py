"""Main multimodal RAG pipeline for DocuMind AI."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, TypedDict

from src.models.llm import LLMClient
from src.parser.image_parser import ImageParser
from src.parser.pdf_parser import PDFParser
from src.retriever.embedding import EmbeddingModel
from src.retriever.faiss_index import FAISSRetriever
from src.verifier.checker import AnswerVerifier

logger = logging.getLogger(__name__)


class EvidencePayload(TypedDict):
    """Typed representation of a retrieved evidence item."""

    chunk_id: str
    text: str
    source: str
    page: Optional[int]
    score: float


class AskPayload(TypedDict):
    """Typed representation of pipeline answer payload."""

    answer: str
    evidence: List[EvidencePayload]
    confidence_score: float
    verification_label: str
    verification_reasoning: str


class DocumentReasoningPipeline:
    """End-to-end pipeline: parse, index, retrieve, answer, and verify."""

    def __init__(self) -> None:
        """Initialize all pipeline components."""
        embedding_model = EmbeddingModel()
        self.pdf_parser = PDFParser()
        self.image_parser = ImageParser()
        self.retriever = FAISSRetriever(embedding_model=embedding_model)
        self.llm = LLMClient()
        self.verifier = AnswerVerifier()

    def ingest_document(self, file_path: str) -> int:
        """Parse and index document chunks; return number of chunks indexed."""
        ext = os.path.splitext(file_path.lower())[1]
        if ext == ".pdf":
            chunks = self.pdf_parser.parse(file_path)
        elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            chunks = self.image_parser.parse(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        self.retriever.build(chunks)
        logger.info("Indexed %d chunks for %s", len(chunks), file_path)
        return len(chunks)

    def ask(self, question: str, top_k: int = 5) -> AskPayload:
        """Run retrieval, answer generation, and verification for a question."""
        results = self.retriever.retrieve(question, top_k=top_k)
        evidence_texts: List[str] = [chunk.text for chunk, _ in results]
        answer = self.llm.generate_answer(question, evidence_texts)
        verification = self.verifier.verify(answer, evidence_texts)

        evidence_payload: List[EvidencePayload] = []
        for chunk, score in results:
            evidence_payload.append(
                {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "score": round(float(score), 5),
                }
            )

        return {
            "answer": answer,
            "evidence": evidence_payload,
            "confidence_score": round(float(verification.confidence_score), 4),
            "verification_label": verification.label,
            "verification_reasoning": verification.reasoning,
        }
