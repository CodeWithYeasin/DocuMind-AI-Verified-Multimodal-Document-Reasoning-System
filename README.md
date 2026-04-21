# documind-ai

DocuMind AI is a production-ready multimodal document reasoning system that ingests PDFs and images, retrieves evidence with FAISS, generates grounded answers, and verifies answer faithfulness.

## Features

- Multimodal parsing for PDF and image files
- Structured chunk generation with metadata
- Embedding generation with sentence-transformers (deterministic fallback included)
- FAISS semantic retrieval (numpy fallback included)
- LLM-based answer generation (OpenAI when configured + local grounded fallback)
- Custom verification module with confidence scoring and labels (`VALID`, `WEAK`, `HALLUCINATED`)
- FastAPI backend with secure upload handling and validation
- Streamlit UI for end-user interaction
- Unit tests for parser, retriever, and verifier

## Project Structure

```text
documind-ai/
├── data/
├── src/
│   ├── parser/
│   ├── retriever/
│   ├── models/
│   ├── verifier/
│   ├── pipeline/
│   └── utils/
├── configs/
├── api/
├── app/
├── tests/
├── notebooks/
├── requirements.txt
├── Dockerfile
├── README.md
├── .env.example
└── .gitignore
```

## Architecture

1. **Parser layer** (`src/parser`): Parses PDF/image documents into structured chunks.
2. **Retriever layer** (`src/retriever`): Encodes chunks and builds a FAISS index for semantic search.
3. **Generation layer** (`src/models`): Produces answers using retrieved context.
4. **Verification layer** (`src/verifier`): Scores grounding confidence and classifies reliability.
5. **Pipeline layer** (`src/pipeline`): Orchestrates parse → retrieve → answer → verify.
6. **Serving/UI layers** (`api`, `app`): Exposes API and web interface.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```

API endpoints:

- `POST /upload` → upload a PDF/image document
- `POST /ask` → ask a question using `document_id`

## Run Streamlit UI

```bash
streamlit run app/app.py
```

Set `API_URL` in `.env` if your API host differs.

## Example Usage

1. Upload document with `/upload`
2. Send question to `/ask` with payload:

```json
{
  "document_id": "<id-from-upload>",
  "question": "What is the contract renewal date?",
  "top_k": 5
}
```

Response includes:

- `answer`
- `evidence` (retrieved chunks)
- `confidence_score`
- `verification_label`

## Security Considerations

- API keys loaded via `.env` (`OPENAI_API_KEY`)
- File upload extension whitelist
- File-size limits (`MAX_UPLOAD_SIZE_MB`)
- Current API session storage is in-memory; use a shared store (e.g., Redis) for multi-worker production deployments
- No secret values committed

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
```
