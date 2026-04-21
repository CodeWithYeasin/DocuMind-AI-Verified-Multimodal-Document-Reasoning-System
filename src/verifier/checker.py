"""Answer grounding verifier for DocuMind AI."""

from __future__ import annotations

import re
from typing import Sequence

from pydantic import BaseModel, Field

from src.utils.schemas import DocumentChunk


class VerificationResult(BaseModel):
    """Verification output with confidence and label."""

    confidence_score: float = Field(ge=0.0, le=1.0)
    label: str
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    overlap_score: float = Field(ge=0.0, le=1.0)


class AnswerVerifier:
    """Verify whether a generated answer is grounded in retrieved evidence."""

    _stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "it",
    }

    def verify(
        self,
        answer: str,
        evidence_chunks: Sequence[DocumentChunk],
        question: str | None = None,
    ) -> VerificationResult:
        """Verify answer grounding using lexical evidence coverage and overlap."""
        answer_tokens = [token for token in self._tokenize(answer) if token not in self._stopwords]
        if not answer_tokens or not evidence_chunks:
            return VerificationResult(
                confidence_score=0.0,
                label="HALLUCINATED",
                evidence_coverage=0.0,
                overlap_score=0.0,
            )

        evidence_text = " ".join(chunk.text for chunk in evidence_chunks)
        evidence_tokens = set(self._tokenize(evidence_text))

        covered = sum(1 for token in answer_tokens if token in evidence_tokens)
        evidence_coverage = covered / max(1, len(answer_tokens))

        overlap_score = max(self._jaccard(answer, chunk.text) for chunk in evidence_chunks)

        question_relevance = 0.0
        if question:
            question_tokens = [token for token in self._tokenize(question) if token not in self._stopwords]
            if question_tokens:
                question_hits = sum(1 for token in question_tokens if token in evidence_tokens)
                question_relevance = question_hits / max(1, len(question_tokens))

        confidence = (0.55 * evidence_coverage) + (0.35 * overlap_score) + (0.10 * question_relevance)
        confidence = max(0.0, min(1.0, confidence))

        if confidence >= 0.75:
            label = "VALID"
        elif confidence >= 0.45:
            label = "WEAK"
        else:
            label = "HALLUCINATED"

        return VerificationResult(
            confidence_score=confidence,
            label=label,
            evidence_coverage=evidence_coverage,
            overlap_score=overlap_score,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric tokens."""
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _jaccard(self, left: str, right: str) -> float:
        """Compute Jaccard overlap score between two texts."""
        left_tokens = set(self._tokenize(left)) - self._stopwords
        right_tokens = set(self._tokenize(right)) - self._stopwords
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
