"""Unit tests for parser modules."""

from pathlib import Path

import fitz
from PIL import Image

from src.parser.image_parser import ImageParser
from src.parser.pdf_parser import PDFParser


def test_pdf_parser_extracts_chunks(tmp_path: Path) -> None:
    """PDF parser should extract at least one chunk from simple PDF."""
    pdf_path = tmp_path / "sample.pdf"

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hello DocuMind PDF parser")
    document.save(pdf_path)
    document.close()

    parser = PDFParser(chunk_size=200, chunk_overlap=20)
    chunks = parser.parse(str(pdf_path), document_id="doc-test")

    assert len(chunks) >= 1
    assert "Hello" in chunks[0].text


def test_image_parser_extracts_chunks_with_mocked_ocr(tmp_path: Path, monkeypatch) -> None:
    """Image parser should return chunks when OCR data is provided."""
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(image_path)

    class MockTesseract:
        """Simple tesseract mock."""

        class Output:
            """Mock output enum."""

            DICT = "DICT"

        @staticmethod
        def image_to_data(*args, **kwargs):
            """Return deterministic OCR output."""
            return {
                "text": ["Hello", "DocuMind"],
                "conf": ["95", "91"],
                "block_num": [1, 1],
                "par_num": [1, 1],
                "line_num": [1, 1],
                "left": [10, 60],
                "top": [10, 10],
                "width": [40, 30],
                "height": [10, 10],
            }

    monkeypatch.setattr("src.parser.image_parser.pytesseract", MockTesseract)

    parser = ImageParser(chunk_size=200, chunk_overlap=20)
    chunks = parser.parse(str(image_path), document_id="img-test")

    assert len(chunks) == 1
    assert "Hello" in chunks[0].text
