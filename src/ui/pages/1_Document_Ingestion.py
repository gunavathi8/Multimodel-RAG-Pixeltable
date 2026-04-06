from __future__ import annotations

import streamlit as st

from src.ui.api_client import api_get, api_post


st.set_page_config(page_title="Document Ingestion", page_icon="📥", layout="wide")
st.title("Document Ingestion")
st.caption("Upload a document and run the full ingestion pipeline required for grounded chat.")
st.page_link("Home.py", label="Back to Home", icon="🏠")
st.page_link("pages/2_Ask_AI.py", label="Go to Ask AI", icon="🤖")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file and st.button("Ingest document", type="primary"):
    with st.spinner("Uploading and processing document..."):
        result = api_post(
            "/documents/ingest",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/pdf")},
        )
    st.success(f"Document processed: {result['document_name']}")
    st.json(result)

st.subheader("Registered documents")
documents = api_get("/documents")
if documents:
    st.dataframe(documents, width="stretch")
else:
    st.info("No documents registered yet.")
