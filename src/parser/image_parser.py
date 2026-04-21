"""Image parsing utilities for OCR and visual metadata extraction."""

from __future__ import annotations

import logging
import uuid
from typing import List

from PIL import Image, ImageStat

from src.utils.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class ImageParser:
    """Parse image files into OCR and visual-summary chunks."""

    def __init__(self, chunk_size: int = 500) -> None:
        """Initialize image parser settings."""
        self.chunk_size = chunk_size

    def parse(self, file_path: str) -> List[DocumentChunk]:
        """Parse image and return OCR and visual description chunks."""
        with Image.open(file_path) as image:
            image = image.convert("RGB")
            summary = self._visual_summary(image, file_path)
            ocr_text = self._extract_text_via_ocr(image)

        chunks: List[DocumentChunk] = [
            DocumentChunk(
                id=f"img-summary-{uuid.uuid4().hex[:8]}",
                source=file_path,
                text=summary,
                metadata={"type": "image_visual"},
            )
        ]

        if ocr_text:
            for index, text_chunk in enumerate(self._split_text(ocr_text)):
                chunks.append(
                    DocumentChunk(
                        id=f"img-ocr-{index}-{uuid.uuid4().hex[:8]}",
                        source=file_path,
                        text=text_chunk,
                        metadata={"type": "image_ocr"},
                    )
                )

        logger.info("Parsed %d chunks from image: %s", len(chunks), file_path)
        return chunks

    @staticmethod
    def _visual_summary(image: Image.Image, file_path: str) -> str:
        """Build a deterministic visual summary string for retrieval context."""
        stats = ImageStat.Stat(image)
        mean_rgb = [round(channel, 2) for channel in stats.mean[:3]]
        width, height = image.size
        return (
            f"Image file {file_path} has resolution {width}x{height}, mode {image.mode}, "
            f"and average RGB values {mean_rgb}."
        )

    @staticmethod
    def _extract_text_via_ocr(image: Image.Image) -> str:
        """Extract OCR text using pytesseract when available."""
        try:
            import pytesseract  # type: ignore
        except ImportError as exc:
            logger.warning("OCR unavailable: %s", exc)
            return ""

        tesseract_not_found_error = getattr(
            pytesseract,
            "TesseractNotFoundError",
            RuntimeError,
        )
        try:
            text = pytesseract.image_to_string(image)
            return text.strip()
        except (tesseract_not_found_error, OSError, RuntimeError, TypeError) as exc:  # type: ignore[misc]
            logger.warning(
                "OCR failed: %s. Install and configure tesseract-ocr to enable image text extraction.",
                exc,
            )
            return ""

    def _split_text(self, text: str) -> List[str]:
        """Split OCR text into fixed-size chunks."""
        words = text.split()
        if not words:
            return []
        chunk_len = max(40, int(self.chunk_size / 5))
        return [" ".join(words[i : i + chunk_len]) for i in range(0, len(words), chunk_len)]
