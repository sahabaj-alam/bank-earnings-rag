"""Query the bank earnings RAG pipeline.

Reads a user question, retrieves top-K relevant chunks from ChromaDB, 
builds a grounded prompt, and calls gpt-4o-mini for the answer.

Run:
    python src/query.py "What did JPMC say about credit risk?"
    python src/query.py --top-k 5 "How did banks describe consumer health?"
    python src/query.py --interactive
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI


# --- Config -----------------------------------------------------------------

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "bank_earnings"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARS = 8000 # safety cap on total context length

SYSTEM_PROMPT = """\
You are a precise financial research assistant analyzing US bank earnings call transcripts from Q1 2026.

Answer the user's question using ONLY the provided transcript excerpts. Rules:

1. Cite every claim by referencing the bank name in [brackets]. Example: "JPMC reported $2.3B in credit costs [jpmc]."
2. If multiple banks discussed the sane topic, compare them concisely.
3. If the excerpts don't contain the answer, say so explicitly - do not guess or use outside knowledge.
4. Keep answers focused. 2-4 short paragraphs is the right length.
5. Use specific numbers and quotes when the excerpts provide them.
6. Do not editorialize or give investment advice.

Banks in the dataset: JPMorgan Chase (jpmc), Goldman Sachs (gs), Morgan Stanley (ms), Bank of America (bac), Wells Fargo (wfc).
"""

# --- Core functions ---------------------------------------------------------


def embed_query(client: OpenAI, text: str) -> list[float]:
    """Embed a single user query."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def retrieve(collection, query_embedding: list[float], top_k: int) -> dict:
    """Retrieve top-K chunks from ChromaDB."""
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results=top_k,
    )
    return results

def build_context(results: dict) -> tuple[str, list[dict]]:
    """Format retrieved chunks into a context block for the LLM.
    Returns (context_string, sources_list)."""
    documents = results["documents"][0] 
    metadatas = results["metadatas"][0] 
    distances = results["distances"][0] 

    context_parts = []
    sources = []
    total_chars = 0

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        bank = meta.get("bank", "unknown")
        quarter = meta.get("quarter", "unknown") 
        chunk_idx = meta.get("chunk_index", -1)

        excerpt = f"[Excerpt {i+1} | bank={bank} | quarter={quarter} chunk={chunk_idx}]\n{doc}\n"

        if total_chars + len(excerpt) > MAX_CONTEXT_CHARS: 
            break

        context_parts.append(excerpt)
        sources.append({
            "rank": i + 1,
            "bank": bank,
            "quarter": quarter,
            "chunk_index": chunk_idx, 
            "distance": round(dist, 3),
            "source_url": meta.get("source_url", ""),
        })
        total_chars += len(excerpt)

    return "\n".join(context_parts), sources


def generate_answer(client: OpenAI, question: str, context: str) -> str:
    """Call gpt-4o-mini with the question + retrieved context."""
    user_message = f"""\
Question: {question}

Relevant transcript excerpts:

{context}

Answer the question using only the excerpts above. Cite banks in [brackets]."""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2, # low temperature for factual grounding
    )
    return resp.choices[0].message.content.strip()

def answer_question(client: OpenAI, collection, question: str, top_k: int) -> dict:
    """Full RAG pipeline for a single question."""
    q_emb = embed_query(client, question)
    results = retrieve(collection, q_emb, top_k)
    context, sources = build_context(results)
    answer = generate_answer(client, question, context)
    return {"question": question, "answer": answer, "sources": sources}


# --- Output formatting ------------------------------------------------------


def print_result(result: dict) -> None:
    """Pretty-print the answer + sources."""
    print()
    print("=" * 72)
    print(f"Q: {result['question']}")
    print("=" * 72)
    print()
    print(textwrap.fill(result["answer"], width=72, 
                        replace_whitespace=False, drop_whitespace=False))
    print()
    print("-" * 72)
    print("Sources:") 
    for s in result["sources"]:
        print(f" [{s['rank']}] {s['bank']} {s['quarter']} chunk={s['chunk_index']} (dist={s['distance']})")
    print()


# --- Main -------------------------------------------------------------------

def main()-> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=None, 
                        help="The question to ask (or use --interactive)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, 
                        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})")
    parser.add_argument("--interactive", action="store_true", 
                        help="Enter interactive REPL mode")
    args = parser.parse_args()

    load_dotenv()

    api_key= os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        return 1

    if not CHROMA_DIR.exists():
        print(f"ERROR: {CHROMA_DIR}/ not found. Run `python src/ingest.py` first.")
        return 1

    client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), 
        settings=Settings(anonymized_telemetry=False), 
    )

    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"ERROR: Collection '{COLLECTION_NAME}' not found. Run ingestion first.")
        return 1

    print(f"Loaded collection '{COLLECTION_NAME}' with {collection.count()} chunks.\n")

    if args.interactive:
        print("Interactive mode. Type a question, or 'exit' to quit.\n")
        while True:
            try:
                question = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit", ""}:
                break 
            result = answer_question(client, collection, question, args.top_k)
            print_result(result)
        return 0

    if not args.question:
        print("ERROR: Provide a question, or use --interactive")
        print("Example: python src/query.py \"What did JPMC say about credit risk?\"")
        return 1
    
    result = answer_question(client, collection, args.question, args.top_k)
    print_result(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())