"""Shared schema objects used across DocuMind AI modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


@dataclass(slots=True)
class DocumentChunk:
    """Represents a parsed chunk from a document source."""

    id: str
    source: str
    text: str
    page: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AskRequest(BaseModel):
    """Request body for asking a question against an uploaded document."""

    document_id: str = Field(min_length=1)
    question: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class UploadResponse(BaseModel):
    """Response returned after document ingestion."""

    document_id: str
    chunks_indexed: int


class EvidenceItem(BaseModel):
    """Evidence entry returned with answer responses."""

    chunk_id: str
    text: str
    source: str
    page: Optional[int]
    score: float


class AskResponse(BaseModel):
    """Question-answer response returned by API."""

    answer: str
    evidence: List[EvidenceItem]
    confidence_score: float = Field(ge=0.0, le=1.0)
    verification_label: str

    @field_validator("verification_label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        """Validate the verification label enum values."""
        valid = {"VALID", "WEAK", "HALLUCINATED"}
        if value not in valid:
            raise ValueError(f"verification_label must be one of {valid}")
        return value
