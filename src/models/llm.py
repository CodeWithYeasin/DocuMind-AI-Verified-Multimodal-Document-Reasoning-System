"""LLM wrapper for grounded answer generation."""

from __future__ import annotations

import logging
import os
import re
from typing import Sequence

from src.utils.schemas import DocumentChunk

LOGGER = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


class LLMService:
    """Generate answers using retrieved context, with OpenAI or extractive fallback."""

    def __init__(self) -> None:
        """Initialize LLM provider based on environment configuration."""
        self.provider = os.getenv("LLM_PROVIDER", "auto").lower()
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = None

        openai_key = os.getenv("OPENAI_API_KEY")
        if self.provider in {"auto", "openai"} and openai_key and OpenAI is not None:
            self.provider = "openai"
            self._client = OpenAI(api_key=openai_key)
            LOGGER.info("Using OpenAI provider with model %s", self.model_name)
        else:
            self.provider = "extractive"
            LOGGER.info("Using extractive fallback provider")

    def generate_answer(self, question: str, evidence_chunks: Sequence[DocumentChunk]) -> str:
        """Generate answer grounded in the provided evidence chunks."""
        if not question.strip():
            raise ValueError("question must be non-empty")
        if not evidence_chunks:
            return "I do not have enough evidence in the uploaded document to answer this question."

        if self.provider == "openai" and self._client is not None:
            try:
                return self._generate_openai(question=question, evidence_chunks=evidence_chunks)
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("OpenAI generation failed, falling back to extractive mode: %s", exc)

        return self._generate_extractive(question=question, evidence_chunks=evidence_chunks)

    def _generate_openai(self, question: str, evidence_chunks: Sequence[DocumentChunk]) -> str:
        """Generate grounded answer using OpenAI chat completions."""
        context = "\n\n".join(
            f"[Chunk {idx + 1}] {chunk.text}" for idx, chunk in enumerate(evidence_chunks)
        )
        prompt = (
            "You are a strict document QA assistant. "
            "Answer only using the evidence context. "
            "If evidence is insufficient, explicitly say so.\n\n"
            f"Question: {question}\n\nEvidence:\n{context}"
        )
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Provide concise, evidence-grounded answers only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip()

    def _generate_extractive(self, question: str, evidence_chunks: Sequence[DocumentChunk]) -> str:
        """Generate deterministic extractive answer from evidence."""
        question_tokens = self._tokenize(question)
        if not question_tokens:
            return "I do not have enough evidence in the uploaded document to answer this question."

        best_sentence = ""
        best_score = -1.0

        for chunk in evidence_chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            for sentence in sentences:
                sentence_tokens = self._tokenize(sentence)
                if not sentence_tokens:
                    continue
                overlap = len(set(question_tokens) & set(sentence_tokens))
                score = overlap / max(1, len(set(question_tokens)))
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()

        if best_sentence and best_score > 0:
            return best_sentence
        return "I do not have enough evidence in the uploaded document to answer this question."

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric tokens."""
        return re.findall(r"[a-zA-Z0-9]+", text.lower())
