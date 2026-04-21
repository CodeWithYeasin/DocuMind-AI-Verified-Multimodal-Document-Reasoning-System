"""Unit tests for grounding verifier module."""

from __future__ import annotations

import unittest

from src.verifier.checker import AnswerVerifier


class VerifierTests(unittest.TestCase):
    """Validate verification confidence and labels."""

    def test_valid_answer_has_high_confidence(self) -> None:
        """Grounded answer should be labeled as VALID or WEAK with good score."""
        verifier = AnswerVerifier()
        answer = "Paris is the capital of France."
        evidence = ["France has a capital city called Paris."]

        result = verifier.verify(answer, evidence)
        self.assertGreaterEqual(result.confidence_score, 0.42)
        self.assertIn(result.label, {"VALID", "WEAK"})

    def test_hallucinated_answer_has_low_confidence(self) -> None:
        """Unsupported answer should be marked hallucinated."""
        verifier = AnswerVerifier()
        answer = "The moon is made of cheese."
        evidence = ["Saturn has many rings composed of ice and rock."]

        result = verifier.verify(answer, evidence)
        self.assertLess(result.confidence_score, 0.42)
        self.assertEqual(result.label, "HALLUCINATED")


if __name__ == "__main__":
    unittest.main()
