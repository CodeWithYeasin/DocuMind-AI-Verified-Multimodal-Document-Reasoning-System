"""PDF parsing utilities with structured chunk output."""

from __future__ import annotations

import logging
import uuid
from typing import List

import pdfplumber

from src.utils.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class PDFParser:
    """Parse PDF files into semantically grouped text chunks."""

    def __init__(self, chunk_size: int = 700, overlap: int = 100) -> None:
        """Initialize parser chunking parameters."""
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse(self, file_path: str) -> List[DocumentChunk]:
        """Parse a PDF file into structured document chunks."""
        chunks: List[DocumentChunk] = []
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                lines = self._extract_lines(page)
                page_text = "\n".join(lines).strip()
                if not page_text:
                    continue
                for index, text_chunk in enumerate(self._split_text(page_text)):
                    chunk_id = f"pdf-{page_number}-{index}-{uuid.uuid4().hex[:8]}"
                    chunks.append(
                        DocumentChunk(
                            id=chunk_id,
                            source=file_path,
                            page=page_number,
                            text=text_chunk,
                            metadata={
                                "type": "pdf_text",
                                "page_width": page.width,
                                "page_height": page.height,
                            },
                        )
                    )

        logger.info("Parsed %d chunks from PDF: %s", len(chunks), file_path)
        return chunks

    @staticmethod
    def _extract_lines(page: pdfplumber.page.Page) -> List[str]:
        """Extract page lines with basic layout handling by y-coordinate grouping."""
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        if not words:
            return []

        line_map: dict[int, list[str]] = {}
        for word in words:
            y_bucket = int(float(word["top"]) / 8)
            line_map.setdefault(y_bucket, []).append(word["text"])

        return [" ".join(line_map[key]).strip() for key in sorted(line_map.keys())]

    def _split_text(self, text: str) -> List[str]:
        """Split text into overlapping word-based chunks."""
        words = text.split()
        if not words:
            return []

        chunk_word_len = max(50, int(self.chunk_size / 5))
        overlap_word_len = max(10, int(self.overlap / 5))
        chunks: List[str] = []

        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_word_len)
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start = end - overlap_word_len
        return chunks
