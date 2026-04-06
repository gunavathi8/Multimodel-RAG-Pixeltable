from __future__ import annotations

import streamlit as st

from src.ui.api_client import api_get, api_post


st.set_page_config(page_title="Ask AI", page_icon="🤖", layout="wide")
st.title("Ask AI")
st.caption("Choose a document, start a chat session, and ask grounded questions answered from retrieved text and visual evidence.")
st.page_link("Home.py", label="Back to Home", icon="🏠")

documents = api_get("/documents")
document_names = [doc["document_name"] for doc in documents]

if not document_names:
    st.info("No documents available yet. Please ingest a document first.")
else:
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = None

    sessions = api_get("/chat/sessions")
    with st.sidebar:
        st.header("Chat sessions")
        if st.button("New chat", width="stretch"):
            st.session_state.active_session_id = None

        if sessions:
            for session in sessions:
                label = f"{session['session_title']} · {session['document_name']}"
                if st.button(label, key=f"session_{session['session_id']}", width="stretch"):
                    st.session_state.active_session_id = session["session_id"]
        else:
            st.caption("No saved sessions yet.")

    selected_document = st.selectbox("Select document for this chat", document_names)
    history_params = {"session_id": st.session_state.active_session_id} if st.session_state.active_session_id else {"document_name": selected_document}
    history = api_get("/chat/history", params=history_params)

    if st.session_state.active_session_id is None:
        st.info("Starting a new chat. Pick a document and ask your first question.")
    elif history:
        selected_document = history[0]["document_name"]
        for item in history:
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.write(item["answer"])
    else:
        st.info("No messages yet in this session.")

    prompt = st.chat_input(f"Ask a question about {selected_document}")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner("Generating grounded answer..."):
            result = api_post(
                "/chat/ask",
                json={
                    "document_name": selected_document,
                    "question": prompt,
                    "session_id": st.session_state.active_session_id,
                    "session_title": prompt[:80] if st.session_state.active_session_id is None else None,
                },
            )
        st.session_state.active_session_id = result["session_id"]
        with st.chat_message("assistant"):
            st.write(result["answer"])
        with st.expander("Show retrieved evidence"):
            st.write("Text context")
            st.write(result["text_context"])
            st.write("Visual context")
            st.write(result["visual_context"])
