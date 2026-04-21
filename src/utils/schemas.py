"""Shared data schemas for DocuMind AI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(slots=True)
class DocumentChunk:
    """Represents a structured chunk extracted from a document."""

    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DocumentChunk":
        """Create a chunk object from dictionary payload."""
        return cls(
            chunk_id=payload["chunk_id"],
            document_id=payload["document_id"],
            text=payload["text"],
            metadata=payload.get("metadata", {}),
        )
