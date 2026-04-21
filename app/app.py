"""Streamlit UI for DocuMind AI."""

from __future__ import annotations

import io
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
MAX_FILE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))

st.set_page_config(page_title="DocuMind AI", layout="wide")
st.title("DocuMind AI: Verified Multimodal Document Reasoning")
st.caption("Upload a document, ask grounded questions, and inspect evidence confidence.")

uploaded = st.file_uploader(
    "Upload PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if uploaded is not None:
    payload = uploaded.getvalue()
    if len(payload) > MAX_FILE_MB * 1024 * 1024:
        st.error(f"File too large. Max allowed is {MAX_FILE_MB} MB.")
    elif st.button("Process Document"):
        files = {
            "file": (uploaded.name, io.BytesIO(payload), uploaded.type or "application/octet-stream")
        }
        with st.spinner("Uploading and indexing document..."):
            response = requests.post(f"{API_URL}/upload", files=files, timeout=120)

        if response.ok:
            data = response.json()
            st.session_state.document_id = data["document_id"]
            st.success(f"Indexed {data['chunks_indexed']} chunks")
        else:
            st.error(f"Upload failed: {response.text}")

question = st.text_input("Ask a question about the document")
if st.button("Ask"):
    if not st.session_state.document_id:
        st.warning("Please upload and process a document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        body = {
            "document_id": st.session_state.document_id,
            "question": question.strip(),
            "top_k": 5,
        }
        with st.spinner("Generating grounded answer..."):
            response = requests.post(f"{API_URL}/ask", json=body, timeout=120)

        if response.ok:
            result = response.json()
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Verification")
            st.metric("Confidence Score", f"{result['confidence_score']:.2f}")
            st.write(f"Label: **{result['verification_label']}**")

            st.subheader("Evidence")
            for idx, item in enumerate(result["evidence"], start=1):
                with st.expander(f"Evidence #{idx} | score={item['score']:.3f}"):
                    st.write(item["text"])
                    st.caption(f"Source: {item['source']} | Page: {item.get('page')}")
        else:
            st.error(f"Question answering failed: {response.text}")
