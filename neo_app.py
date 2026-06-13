# neo_app.py
# Neo — AI-powered document intelligence
# A RAG-based Q&A interface for technical documentation
# Built by Tyrelle Newton | Portfolio Project 2026

import os
import streamlit as st
from dotenv import load_dotenv

from phase2_document_loader import load_pdf
from phase3_chunker import chunk_pages
from phase4_embeddings import build_vector_store, search
from phase5_rag_pipeline import ask

# --- Load environment variables ---
load_dotenv()

# --- Page configuration ---
# This must be the first Streamlit command in the file
st.set_page_config(
    page_title="Neo — Document Intelligence",
    page_icon="🕶️",
    layout="wide"
)

# --- Header ---
st.markdown("# 🕶️ Neo")
st.markdown("#### AI-powered document intelligence")
st.markdown("Ask anything about your loaded technical documentation. Answers are grounded in the source material and cited by page.")
st.divider()


# --- Load and index documents ---
# st.cache_resource tells Streamlit to run this function ONCE and reuse the result
# Without this, the PDF would reload and re-embed on every single interaction
# This is the Streamlit equivalent of loading at server startup in Flask
@st.cache_resource
def initialise_pipeline():
    """
    Loads the FIA Technical Regulations, chunks them, and builds the vector store.
    Cached so this only runs once per session — not on every question.
    """
    filepath = os.path.join(
        "documents",
        "fia_2026_f1_regulations_-_section_c_technical_-_iss_18_-_2026-05-07.pdf"
    )

    pages = load_pdf(filepath)
    chunks = chunk_pages(pages)
    index, embedding_model = build_vector_store(chunks)

    return pages, chunks, index, embedding_model


# --- Sidebar — document status ---
with st.sidebar:
    st.markdown("### 📁 Loaded documents")

    # Show a spinner while the pipeline initialises on first load
    with st.spinner("Loading and indexing documents..."):
        pages, chunks, index, embedding_model = initialise_pipeline()

    # Once loaded, show document status
    st.success("Documents ready")
    st.markdown(f"**Document:** FIA 2026 F1 Technical Regulations")
    st.markdown(f"**Pages:** {len(pages)}")
    st.markdown(f"**Chunks indexed:** {len(chunks)}")
    st.divider()
    st.markdown("### 💡 Example questions")
    st.markdown("- What are the front wing dimension limits?")
    st.markdown("- What are the power unit energy store rules?")
    st.markdown("- What is the maximum fuel flow rate?")
    st.markdown("- What are the crash structure requirements?")
    st.divider()
    st.caption("Built by Tyrelle Newton · 2026")


# --- Chat interface ---
# st.session_state persists data between interactions in Streamlit
# Without it, the chat history would reset every time the user asks a question
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "🕶️ Hi, I'm Neo. How may I help you today? Ask me anything about the FIA 2026 Technical Regulations and I'll find the answer from the source documents.",
            "sources": []
        }
    ]
# --- Display existing chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources under assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.caption(source)


# --- Chat input ---
# st.chat_input creates the message box at the bottom of the screen
if question := st.chat_input("Ask a question about the FIA Technical Regulations..."):

    # --- Display the user's question ---
    with st.chat_message("user"):
        st.markdown(question)

    # --- Add to chat history ---
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # --- Generate and display the answer ---
    with st.chat_message("assistant"):
        with st.spinner("Searching regulations and generating answer..."):
            result = ask(question, index, embedding_model, chunks)

        st.markdown(result["answer"])

        # Show sources in an expandable section
        with st.expander("📄 Sources"):
            for source in result["sources"]:
                st.caption(source)

    # --- Add answer to chat history ---
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })