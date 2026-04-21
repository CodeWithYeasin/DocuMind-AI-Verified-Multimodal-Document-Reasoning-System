"""FastAPI service for DocuMind AI."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline.main_pipeline import DocumentReasoningPipeline

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

pipeline = DocumentReasoningPipeline(data_dir=str(DATA_DIR))


class UploadResponse(BaseModel):
    """Response payload for upload endpoint."""

    document_id: str
    filename: str
    chunks_indexed: int


class AskRequest(BaseModel):
    """Request payload for question-answering endpoint."""

    question: str = Field(min_length=1, max_length=2000)
    document_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=10)


class EvidenceItem(BaseModel):
    """One evidence item in answer payload."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict


class AskResponse(BaseModel):
    """Response payload for question-answering endpoint."""

    answer: str
    evidence: List[EvidenceItem]
    confidence_score: float
    verification_label: str


app = FastAPI(title="DocuMind AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload and index a PDF/image document."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds size limit (10MB)")

    document_id = str(uuid.uuid4())
    safe_name = f"{document_id}{suffix}"
    file_path = UPLOADS_DIR / safe_name
    file_path.write_bytes(content)

    try:
        chunk_count = pipeline.ingest_document(file_path=str(file_path), document_id=document_id)
    except Exception as exc:
        LOGGER.exception("Failed to ingest document")
        raise HTTPException(status_code=400, detail=f"Failed to parse document: {exc}") from exc

    return UploadResponse(document_id=document_id, filename=file.filename or safe_name, chunks_indexed=chunk_count)


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    """Ask a question against indexed documents."""
    try:
        result = pipeline.ask(
            question=payload.question,
            top_k=payload.top_k,
            document_id=payload.document_id,
        )
    except Exception as exc:
        LOGGER.exception("Failed to answer question")
        raise HTTPException(status_code=400, detail=f"Failed to answer question: {exc}") from exc

    return AskResponse(
        answer=result["answer"],
        evidence=result["evidence"],
        confidence_score=result["confidence_score"],
        verification_label=result["verification_label"],
    )
