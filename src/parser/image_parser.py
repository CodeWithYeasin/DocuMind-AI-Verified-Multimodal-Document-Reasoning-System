"""Image parsing module with OCR and basic layout-aware chunk extraction."""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List

from PIL import Image

from src.utils.schemas import DocumentChunk

LOGGER = logging.getLogger(__name__)

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


class ImageParser:
    """Parse image files into structured OCR text chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        """Initialize parser with chunking parameters."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, file_path: str, document_id: str | None = None) -> List[DocumentChunk]:
        """Parse an image into OCR-based structured chunks.

        Args:
            file_path: Absolute path to image file.
            document_id: Optional pre-assigned document ID.

        Returns:
            A list of text chunks with layout metadata.
        """
        if pytesseract is None:
            raise RuntimeError("pytesseract is required for image parsing")

        effective_document_id = document_id or str(uuid.uuid4())
        LOGGER.info("Parsing image file: %s", file_path)

        image = Image.open(file_path).convert("RGB")
        width, height = image.size
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        lines = self._collect_lines(ocr_data)

        chunks: List[DocumentChunk] = []
        text_blob = "\n".join(entry["text"] for entry in lines)
        start = 0
        index = 0

        while start < len(text_blob):
            end = min(len(text_blob), start + self.chunk_size)
            text = text_blob[start:end].strip()
            if text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{effective_document_id}:img:c{index}",
                        document_id=effective_document_id,
                        text=text,
                        metadata={"source": "image", "page": 1, "image_size": {"width": width, "height": height}},
                    )
                )
                index += 1

            if end == len(text_blob):
                break
            start = max(0, end - self.chunk_overlap)

        LOGGER.info("Parsed %d chunks from image", len(chunks))
        return chunks

    def _collect_lines(self, ocr_data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Group OCR word boxes into line-level structured text records."""
        grouped: Dict[tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
        total = len(ocr_data.get("text", []))

        for idx in range(total):
            text = re.sub(r"\s+", " ", str(ocr_data["text"][idx]).strip())
            conf_raw = str(ocr_data.get("conf", ["0"] * total)[idx]).strip()
            try:
                conf = float(conf_raw)
            except ValueError:
                conf = -1.0
            if not text or conf <= 0:
                continue

            key = (
                int(ocr_data.get("block_num", [0] * total)[idx]),
                int(ocr_data.get("par_num", [0] * total)[idx]),
                int(ocr_data.get("line_num", [0] * total)[idx]),
            )
            grouped[key].append(
                {
                    "text": text,
                    "left": int(ocr_data.get("left", [0] * total)[idx]),
                    "top": int(ocr_data.get("top", [0] * total)[idx]),
                    "width": int(ocr_data.get("width", [0] * total)[idx]),
                    "height": int(ocr_data.get("height", [0] * total)[idx]),
                }
            )

        lines: List[Dict[str, Any]] = []
        for _, words in sorted(grouped.items(), key=lambda entry: (entry[0][0], entry[0][1], entry[0][2])):
            words.sort(key=lambda item: item["left"])
            text = " ".join(word["text"] for word in words)
            lines.append({"text": text})

        return lines
