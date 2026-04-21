"""Streamlit UI for DocuMind AI."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("DOCUMIND_API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")
st.title("DocuMind AI: Verified Multimodal Document Reasoning")

if "document_id" not in st.session_state:
    st.session_state.document_id = None

st.subheader("1) Upload document")
uploaded_file = st.file_uploader("Upload PDF or image", type=["pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])

if st.button("Upload", type="primary"):
    if uploaded_file is None:
        st.warning("Please select a file first.")
    else:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
        response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
        if response.ok:
            payload = response.json()
            st.session_state.document_id = payload["document_id"]
            st.success(
                f"Uploaded: {payload['filename']} | Document ID: {payload['document_id']} | Chunks: {payload['chunks_indexed']}"
            )
        else:
            st.error(response.text)

st.subheader("2) Ask question")
question = st.text_input("Enter your question")
top_k = st.slider("Top-K evidence", min_value=1, max_value=10, value=5)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        payload = {
            "question": question,
            "top_k": top_k,
            "document_id": st.session_state.document_id,
        }
        response = requests.post(f"{API_URL}/ask", json=payload, timeout=120)
        if response.ok:
            result = response.json()
            st.markdown("### Answer")
            st.write(result["answer"])

            st.markdown("### Verification")
            st.metric("Confidence Score", f"{result['confidence_score']:.2f}")
            st.write(f"Label: **{result['verification_label']}**")

            st.markdown("### Evidence")
            for idx, item in enumerate(result["evidence"], start=1):
                with st.expander(f"Evidence #{idx} (score: {item['score']:.3f})"):
                    st.write(item["text"])
                    st.json(item["metadata"])
        else:
            st.error(response.text)
