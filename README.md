# DocuMind AI: Verified Multimodal Document Reasoning System

DocuMind AI is a production-ready multimodal RAG system that ingests PDF and image documents, retrieves relevant evidence, generates grounded answers, and verifies answer faithfulness with a custom confidence-scoring module.

## Features

- Multimodal document parsing (PDF + images)
- Layout-aware chunking for retrieval quality
- Dense embedding generation via `sentence-transformers`
- FAISS vector indexing and top-k evidence retrieval
- LLM answer generation (OpenAI or extractive fallback)
- Custom verification module with confidence + labels:
  - `VALID`
  - `WEAK`
  - `HALLUCINATED`
- FastAPI backend (`/upload`, `/ask`)
- Streamlit user interface
- Unit tests for parser, retriever, verifier

## Architecture

```text
Document (PDF/Image)
   -> Parser (text + layout chunks)
   -> Embedding Service
   -> FAISS Retriever (top-k evidence)
   -> LLM Service (grounded answer)
   -> Verifier (confidence + label)
   -> API/UI response
```

## Project Structure

```text
documind-ai/
├── data/
├── src/
│   ├── parser/
│   │   ├── pdf_parser.py
│   │   └── image_parser.py
│   ├── retriever/
│   │   ├── embedding.py
│   │   └── faiss_index.py
│   ├── models/
│   │   └── llm.py
│   ├── verifier/
│   │   └── checker.py
│   ├── pipeline/
│   │   └── main_pipeline.py
│   └── utils/
│       └── schemas.py
├── configs/
├── api/
│   └── main.py
├── app/
│   └── app.py
├── tests/
├── notebooks/
├── requirements.txt
├── Dockerfile
├── README.md
├── .env.example
└── .gitignore
```

## Setup

1. Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment:

```bash
cp .env.example .env
```

If using OpenAI, set `OPENAI_API_KEY` in `.env`.

## Run FastAPI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open docs at: `http://localhost:8000/docs`

## Run Streamlit UI

```bash
streamlit run app/app.py
```

## API Usage

### Upload document

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/absolute/path/to/document.pdf"
```

### Ask question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total revenue?", "document_id": "<DOCUMENT_ID>", "top_k": 5}'
```

Response fields include:

- `answer`
- `evidence`
- `confidence_score`
- `verification_label`

## Security

- API keys loaded from `.env`
- Upload type validation
- File size cap (10MB)
- No secrets committed

## Testing

```bash
pytest -q
```
