"""Unit tests for parser components."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from src.parser.image_parser import ImageParser
from src.parser.pdf_parser import PDFParser


class ParserTests(unittest.TestCase):
    """Validate PDF and image parser behaviors."""

    def test_pdf_parser_extracts_chunks(self) -> None:
        """PDF parser should return non-empty chunks for text PDF."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "DocuMind parser test content for PDF extraction.")
            doc.save(pdf_path)
            doc.close()

            parser = PDFParser()
            chunks = parser.parse(str(pdf_path))

        self.assertGreater(len(chunks), 0)
        joined = " ".join(chunk.text for chunk in chunks)
        self.assertIn("DocuMind", joined)

    def test_image_parser_returns_visual_chunk(self) -> None:
        """Image parser should at least return visual summary chunk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            image = Image.new("RGB", (300, 100), color=(255, 255, 255))
            drawer = ImageDraw.Draw(image)
            drawer.text((10, 40), "DocuMind", fill=(0, 0, 0))
            image.save(image_path)

            parser = ImageParser()
            chunks = parser.parse(str(image_path))

        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("resolution", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
