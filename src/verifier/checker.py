"""Verification module for grounding confidence and hallucination labeling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class VerificationResult:
    """Result of answer grounding verification."""

    confidence_score: float
    label: str
    reasoning: str


class AnswerVerifier:
    """Verify how well an answer is grounded in retrieved evidence."""

    def verify(self, answer: str, evidence_texts: Iterable[str]) -> VerificationResult:
        """Return confidence score and grounding label for an answer."""
        evidence_text = "\n".join(chunk for chunk in evidence_texts if chunk).strip()
        answer_tokens = self._tokens(answer)
        evidence_tokens = self._tokens(evidence_text)

        if not answer_tokens:
            return VerificationResult(0.0, "HALLUCINATED", "Empty answer has no grounding.")
        if not evidence_tokens:
            return VerificationResult(0.0, "HALLUCINATED", "No evidence was supplied.")

        token_coverage = len(answer_tokens.intersection(evidence_tokens)) / len(answer_tokens)
        sentence_support = self._sentence_support(answer, evidence_text)
        confidence = max(0.0, min(1.0, 0.65 * token_coverage + 0.35 * sentence_support))

        if confidence >= 0.72:
            label = "VALID"
        elif confidence >= 0.42:
            label = "WEAK"
        else:
            label = "HALLUCINATED"

        reasoning = (
            f"token_coverage={token_coverage:.2f}, sentence_support={sentence_support:.2f}, "
            f"combined={confidence:.2f}"
        )
        return VerificationResult(confidence, label, reasoning)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """Extract lowercase alphanumeric tokens."""
        return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 1}

    def _sentence_support(self, answer: str, evidence: str) -> float:
        """Estimate support by sentence-level lexical overlap with evidence."""
        sentences = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", answer) if seg.strip()]
        if not sentences:
            return 0.0

        evidence_tokens = self._tokens(evidence)
        supports: List[float] = []
        for sentence in sentences:
            sent_tokens = self._tokens(sentence)
            if not sent_tokens:
                continue
            overlap = len(sent_tokens.intersection(evidence_tokens)) / len(sent_tokens)
            supports.append(overlap)

        if not supports:
            return 0.0
        return sum(supports) / len(supports)
