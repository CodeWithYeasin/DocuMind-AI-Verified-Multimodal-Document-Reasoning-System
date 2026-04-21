"""FastAPI service for DocuMind AI document QA."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from configs.settings import settings
from src.pipeline.main_pipeline import DocumentReasoningPipeline
from src.utils.schemas import AskRequest, AskResponse, UploadResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocuMind AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(settings.data_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DOCUMENT_PIPELINES: Dict[str, DocumentReasoningPipeline] = {}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload and ingest a document for question answering."""
    filename = file.filename or ""
    extension = os.path.splitext(filename.lower())[1]
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds size limit")

    document_id = uuid.uuid4().hex
    safe_name = f"{document_id}{extension}"
    file_path = UPLOAD_DIR / safe_name
    file_path.write_bytes(data)

    pipeline = DocumentReasoningPipeline()
    try:
        chunks = pipeline.ingest_document(str(file_path))
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}") from exc

    DOCUMENT_PIPELINES[document_id] = pipeline
    return UploadResponse(document_id=document_id, chunks_indexed=chunks)


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Answer a question using previously uploaded document context."""
    pipeline = DOCUMENT_PIPELINES.get(request.document_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        result = pipeline.ask(request.question, top_k=request.top_k)
    except Exception as exc:
        logger.exception("Question answering failed")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}") from exc

    return AskResponse(**result)
