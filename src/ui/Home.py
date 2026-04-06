from __future__ import annotations

import streamlit as st

from src.ui.api_client import api_get


def _hide_sidebar() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
        .hero {padding: 2rem 0 1rem 0;}
        .hero h1 {font-size: 3rem; margin-bottom: 0.4rem;}
        .hero p {font-size: 1.05rem; max-width: 52rem; color: #b8c0cc;}
        .pill-row {display: flex; gap: 0.75rem; margin: 1.25rem 0 1.5rem 0; flex-wrap: wrap;}
        .pill {padding: 0.6rem 1rem; border: 1px solid rgba(255,255,255,0.12); border-radius: 999px; background: rgba(255,255,255,0.03); font-size: 0.95rem;}
        .feature-card {border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 1.2rem; height: 100%; background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));}
        .flow-box {border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 1rem 1.2rem; background: #11161d;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Multimodal RAG", page_icon="📄", layout="wide")
    _hide_sidebar()

    st.markdown(
        """
        <div class="hero">
            <h1>Multimodal RAG for real-world PDFs</h1>
            <p>
                Ingest reports, research papers, charts, tables, and image-heavy documents into a grounded AI system that
                answers from both text chunks and visual evidence. Pixeltable powers the multimodal storage, derived views,
                computed pipelines, and retrieval state behind the scenes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_left, nav_right = st.columns(2)
    with nav_left:
        st.page_link("pages/1_Document_Ingestion.py", label="Document Ingestion", icon="📥")
    with nav_right:
        st.page_link("pages/2_Ask_AI.py", label="Ask AI", icon="🤖")

    st.markdown(
        """
        <div class="pill-row">
            <div class="pill">Text chunk retrieval</div>
            <div class="pill">Page-level visual understanding</div>
            <div class="pill">Hybrid evidence search</div>
            <div class="pill">Grounded document chat</div>
            <div class="pill">Pixeltable-native pipelines</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        health = api_get("/health")
        st.success(f"System status: {health['status']}")
    except Exception as exc:
        st.error(f"Backend unavailable: {exc}")

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="feature-card">
                <h3>What this app does</h3>
                <p>Every uploaded PDF is registered once, split into text chunks, converted into page images,
                enriched with page-level vision descriptions, indexed for retrieval, and then used in grounded chat.</p>
                <p>This keeps charts, tables, figures, and layout-heavy pages in play instead of losing them during text-only extraction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="feature-card">
                <h3>Why Pixeltable matters</h3>
                <p>Pixeltable gives the backend one persistent system for source documents, derived multimodal views,
                computed columns, embedding indexes, and QA history. That makes the whole pipeline inspectable and reproducible.</p>
                <p>Instead of stitching together multiple services, the app treats multimodal document state as first-class tables and views.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Architecture flow")
    st.markdown(
        """
        <div class="flow-box">
        Upload PDF → Register in Pixeltable → Create text chunks → Extract page images → Generate vision descriptions →
        Build embedding indexes → Retrieve text + visual evidence → Generate grounded answer in Ask AI
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("System view")
    st.markdown(
        """
        ```text
        Streamlit product UI
           ├─ Document Ingestion
           └─ Ask AI
                 │
                 ▼
             FastAPI backend
                 │
                 ▼
              Pixeltable
           ├─ documents
           ├─ text_chunks
           ├─ page_images
           └─ qa_sessions
                 │
                 ├─ OpenRouter vision/chat models
                 └─ Sentence-transformer embeddings
        ```
        """
    )


if __name__ == "__main__":
    main()
