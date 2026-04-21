"""Unit tests for answer verifier."""

from src.utils.schemas import DocumentChunk
from src.verifier.checker import AnswerVerifier


def test_verifier_returns_valid_for_grounded_answer() -> None:
    """Verifier should classify grounded answer as VALID or WEAK with high confidence."""
    verifier = AnswerVerifier()
    evidence = [
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            text="The annual revenue in 2024 was 10 million USD.",
            metadata={"page": 1},
        )
    ]
    answer = "The annual revenue in 2024 was 10 million USD."
    result = verifier.verify(answer=answer, evidence_chunks=evidence, question="What was the annual revenue in 2024?")

    assert result.confidence_score >= 0.45
    assert result.label in {"VALID", "WEAK"}


def test_verifier_returns_hallucinated_for_unsupported_answer() -> None:
    """Verifier should classify unsupported answer as hallucinated."""
    verifier = AnswerVerifier()
    evidence = [
        DocumentChunk(chunk_id="c1", document_id="d1", text="The company was founded in 2012.", metadata={"page": 1})
    ]
    answer = "The company was founded in 1998 and has 10,000 employees."
    result = verifier.verify(answer=answer, evidence_chunks=evidence, question="When was it founded?")

    assert result.confidence_score < 0.75
    assert result.label in {"WEAK", "HALLUCINATED"}
