"""PDF parsing module with basic layout-aware chunk extraction."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Sequence, Tuple

import fitz

from src.utils.schemas import DocumentChunk

LOGGER = logging.getLogger(__name__)


class PDFParser:
    """Parse PDF files into structured text chunks."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        """Initialize parser with chunking parameters.

        Args:
            chunk_size: Maximum character length per chunk.
            chunk_overlap: Character overlap between adjacent chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, file_path: str, document_id: str | None = None) -> List[DocumentChunk]:
        """Parse a PDF document and return structured chunks.

        Args:
            file_path: Absolute path to PDF file.
            document_id: Optional pre-assigned document ID.

        Returns:
            A list of structured text chunks with page and layout metadata.
        """
        effective_document_id = document_id or str(uuid.uuid4())
        chunks: List[DocumentChunk] = []

        LOGGER.info("Parsing PDF file: %s", file_path)
        with fitz.open(file_path) as document:
            for page_index, page in enumerate(document, start=1):
                blocks = page.get_text("blocks")
                layout_blocks = self._extract_layout_blocks(blocks)
                page_chunks = self._chunk_layout_blocks(
                    layout_blocks=layout_blocks,
                    page_number=page_index,
                    document_id=effective_document_id,
                )
                chunks.extend(page_chunks)

        LOGGER.info("Parsed %d chunks from PDF", len(chunks))
        return chunks

    def _extract_layout_blocks(
        self,
        blocks: Sequence[Tuple[Any, ...]],
    ) -> List[Dict[str, Any]]:
        """Normalize and sort layout blocks from PyMuPDF output."""
        normalized: List[Dict[str, Any]] = []

        for block in blocks:
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            cleaned_text = re.sub(r"\s+", " ", (text or "").strip())
            if not cleaned_text:
                continue
            normalized.append(
                {
                    "text": cleaned_text,
                    "bbox": {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)},
                }
            )

        normalized.sort(key=lambda item: (item["bbox"]["y0"], item["bbox"]["x0"]))
        return normalized

    def _chunk_layout_blocks(
        self,
        layout_blocks: Sequence[Dict[str, Any]],
        page_number: int,
        document_id: str,
    ) -> List[DocumentChunk]:
        """Create overlapping chunks while preserving coarse layout metadata."""
        if not layout_blocks:
            return []

        combined_text = "\n".join(block["text"] for block in layout_blocks)
        chunks: List[DocumentChunk] = []
        start = 0
        index = 0

        while start < len(combined_text):
            end = min(len(combined_text), start + self.chunk_size)
            text = combined_text[start:end].strip()
            if text:
                bbox = {
                    "x0": min(block["bbox"]["x0"] for block in layout_blocks),
                    "y0": min(block["bbox"]["y0"] for block in layout_blocks),
                    "x1": max(block["bbox"]["x1"] for block in layout_blocks),
                    "y1": max(block["bbox"]["y1"] for block in layout_blocks),
                }
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}:p{page_number}:c{index}",
                        document_id=document_id,
                        text=text,
                        metadata={"page": page_number, "bbox": bbox, "source": "pdf"},
                    )
                )
                index += 1

            if end == len(combined_text):
                break
            start = max(0, end - self.chunk_overlap)

        return chunks
