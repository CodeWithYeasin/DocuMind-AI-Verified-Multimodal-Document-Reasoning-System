"""LLM wrapper for answer generation from retrieved contexts."""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable, List

logger = logging.getLogger(__name__)


class LLMClient:
    """Generate answers using OpenAI when configured, else local grounded fallback."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        """Initialize LLM backend and runtime options."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_context_chars = 6000
        self._openai_client = None

        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            try:
                from openai import OpenAI  # type: ignore

                self._openai_client = OpenAI(api_key=api_key)
            except ImportError as exc:
                logger.warning("OpenAI SDK unavailable, using local fallback: %s", exc)

    def generate_answer(self, question: str, contexts: Iterable[str]) -> str:
        """Generate answer grounded in provided contexts."""
        context_list = [text.strip() for text in contexts if text.strip()]
        if not context_list:
            return "No supporting context is available for this question."

        if self._openai_client is not None:
            return self._generate_with_openai(question, context_list)
        return self._generate_locally(question, context_list)

    def _generate_with_openai(self, question: str, contexts: List[str]) -> str:
        """Generate answer using OpenAI chat completion."""
        context = "\n\n".join(contexts)[: self.max_context_chars]
        prompt = (
            "You are a document QA system. Answer using only the provided context. "
            "If uncertain, explicitly say the answer is not in the context.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        response = self._openai_client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def _generate_locally(self, question: str, contexts: List[str]) -> str:
        """Generate deterministic local answer by selecting best-supported sentences."""
        query_tokens = self._tokens(question)
        ranked_sentences: List[tuple[float, str]] = []

        for context in contexts:
            for sentence in re.split(r"(?<=[.!?])\s+", context):
                cleaned = sentence.strip()
                if not cleaned:
                    continue
                score = self._overlap_score(query_tokens, self._tokens(cleaned))
                ranked_sentences.append((score, cleaned))

        ranked_sentences.sort(key=lambda item: item[0], reverse=True)
        best = [text for score, text in ranked_sentences[:3] if score > 0]

        if not best:
            return "The answer could not be grounded confidently in the provided context."
        return " ".join(best)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """Tokenize text to lowercase alphanumeric terms."""
        return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 1}

    @staticmethod
    def _overlap_score(question_tokens: set[str], sentence_tokens: set[str]) -> float:
        """Compute lexical overlap score between query and sentence tokens."""
        if not question_tokens or not sentence_tokens:
            return 0.0
        overlap = len(question_tokens.intersection(sentence_tokens))
        return overlap / max(1, len(question_tokens))
