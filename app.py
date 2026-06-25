"""Streamlit UI for the Bank Earnings RAG pipeline.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
import streamlit as st 
from chromadb.config import Settings 
from dotenv import load_dotenv 
from openai import OpenAI

# Import the core RAG functions we already built 
from src.query import (
    CHAT_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBED_MODEL,
    answer_question,
)

# --- Page setup ------------------------------------------------------

st.set_page_config(
    page_title="Bank Earnings RAG - Q1 2026",
    page_icon="🏦",
    layout="centered"
)

# --- Cached resources ------------------------------------------------

@st.cache_resource
def get_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY not set. Add it to .env locally or to Streamlit secrets in deployment.")
        st.stop()
    return OpenAI(api_key=api_key)


@st.cache_resource
def get_collection():
    if not Path(CHROMA_DIR).exists():
        st.error(f"ChromaDB not found at {CHROMA_DIR}/. Run `python src/ingest.py` first.")
        st.stop()
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), 
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        return chroma_client.get_collection (COLLECTION_NAME)
    except Exception:
        st.error(f"Collection '{COLLECTION_NAME}' not found. Run ingestion first.")
        st.stop()


# --- UI --------------------------------------------------------------

st.title("🏦 Bank Earnings RAG")
st.caption(
    "Ask questions about Q1 2026 earnings calls from JPMorgan Chase, Goldman Sachs, "
    "Morgan Stanley, Bank of America, and Wells Fargo. Answers are grounded in "
    "publicly available transcripts and cite sources."
)

with st.expander("⚠️ Disclaimer", expanded=False):
    st.markdown(
        """
        This is a personal portfolio project for educational purposes only.
        - **Not affiliated with any bank or financial institution.**
        - **Not investment advice.** Do not use answers for financial decisions.
        - Built with publicly available earnings call transcripts.
        - Answers may contain errors. Always verify against original sources.
        """
    )

#Load resources (cached)
client = get_openai_client()
collection = get_collection()

st.markdown(f"**Indexed:** {collection.count()} chunks across 5 banks • Model: `{CHAT_MODEL}`")

#Example questions to seed the input
EXAMPLES = [
    "How did major banks describe consumer financial health?",
    "What did JPMC say about credit risk and loan loss provisions?",
    "Compare net interest income trends across the banks.",
    "Which banks discussed AI investments or technology spending?",
    "What were the main concerns analysts raised during Q&A?",
]

st.markdown("**Try a sample question:**")
cols = st.columns (len(EXAMPLES))
selected_example = None
for i, ex in enumerate (EXAMPLES):
    if cols[i].button(f"{i+1}", help=ex, use_container_width=True):
        selected_example = ex

# Initialize session state for the question
if "question_text" not in st.session_state:
    st.session_state.question_text = ""

# If user clicked an example button, update session state
if selected_example:
    st.session_state.question_text = selected_example

#Question input - bound to session state via key
question = st.text_area(
    "Your question:",
    value=st.session_state.question_text,
    height=80,
    placeholder="e.g. What did Jamie Dimon say about the economic outlook?",
    key="question_input"
)

col_a, col_b = st.columns([1, 4])
with col_a:
    top_k = st.number_input("Top K chunks", min_value=1, max_value=10, 
                            value=DEFAULT_TOP_K, step=1)
with col_b:
    submit = st.button("🔍 Ask", type="primary", use_container_width=True)

# --- Run RAG ---------------------------------------------------------

if submit:
    # Read the current text_area value (most recent user edit wins)
    current_q = question.strip() if question else ""

    if not current_q:
        st.warning("Please enter a question")
    else:
        with st.spinner("Retrieving relevant excerpts and generating answer..."):
            try:
                result = answer_question(client, collection, question.strip(), int(top_k))
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        
        st.markdown("### Answer")
        st.markdown(result["answer"])

        with st.expander (f"📚 Sources ({len(result['sources'])} excerpts used)", expanded=False):
            for s in result["sources"]:
                st.markdown(
                    f"**[{s['rank']}]** `{s['bank']}` • {s['quarter']} • "
                    f"chunk={s['chunk_index']} • distance={s['distance']}"
                )
                if s.get("source_url"):
                    st.caption(s["source_url"])

# --- Footer ----------------------------------------------------------

st.divider()
st.caption(
    "Built with OpenAI • ChromaDB • Streamlit • "
    "[Source on GitHub](https://github.com/sahabaj-alam/bank-earnings-rag) • MIT License"
)