"""Ingest bank earnings transcripts into ChromaDB.

Reads all .txt files from data/, chunks thes with tiktoken, embeds each 
chunk via OpenAI text-embedding-3-small, and stores them in a local 
ChromaDB collection at ./chroma_db/.

Run:
    python src/ingest.py                #full run (calls OpenAI, -50.001)
    python src/ingest.py -dry-run        #chunk only, no API/DB calls
    python src/ingest.py --reset        #wipe collection and re-ingest
"""

from __future__ import annotations

import argparse
import os
import sys
import time 
from pathlib import Path

import chromadb
import tiktoken
from chromadb.config import Settings 
from dotenv import load_dotenv
from openai import OpenAI

# --- Config -------------------------------------------------------------------

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db") 
COLLECTION_NAME = "bank_earnings"

EMBED_MODEL = "text-embedding-3-small" 
EMBED_BATCH_SIZE = 100 # OpenAI accepts up to ~2048; 100 keeps things safe

CHUNK_TOKENS = 500
CHUNK_OVERLAP = 50
TOKENIZER = tiktoken.get_encoding("cl100k_base") # used by gpt-4o-mini embeddings

# --- Helpers -------------------------------------------------------------------

def parse_metadata(text: str, filename: str) -> tuple[dict, str]:
    """Extract attribution header (lines starting with #) and strip it 
    from the body. Returns (metadata_dict, body_text)."""
    lines = text.splitlines()
    header_lines = []
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            header_lines.append(line)
        elif line.strip() == "" and header_lines:
            # blank line after header marks end of header block
            body_start = i + 1
            break 
        else:
            body_start = i
            break

    body = "\n".join(lines [body_start:]).strip()

    # Derive bank + quarter from filename: e.g. jpmc_q1_2026.txt
    stem = Path(filename).stem # jpmc_q1_2026
    parts = stem.split("_")
    bank = parts[0] if parts else "unknown"
    quarter = "_".join(parts[1:]) if len(parts) > 1 else "unknown"

    source_line = next((h for h in header_lines if "Source:" in h), "") 
    source_url = source_line.replace("# Source:", "").strip() if source_line else ""

    metadata = {
        "bank": bank,
        "quarter": quarter,
        "source_file": filename,
        "source_url": source_url,
    }
    return metadata, body

def chunk_text(text: str, max_tokens: int = CHUNK_TOKENS, 
               overlap: int = CHUNK_OVERLAP) -> list[str]: 
    """Split text into overlapping token windows."""
    tokens = TOKENIZER.encode(text)
    chunks = []
    start = 0
    step = max_tokens - overlap
    while start < len(tokens): 
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end] 
        chunks.append(TOKENIZER.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += step
    return chunks

def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """---Embed a batch of texts via OpenAI. Retries once on transient error."""
    for attempt in (1, 2):
        try: 
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            if attempt == 2:
                raise
            print(f" embed retry after error: {e}")
            time.sleep(2)
    return [] # unreachable

# --- Main -------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__) 
    parser.add_argument("--dry-run", action="store_true", 
                        help="Chunk only, no API calls, no DB writes.")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing collection before ingesting.") 
    args = parser.parse_args()

    load_dotenv()

    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR}/ not found. Run from project root.")
        return 1

    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        print(f"ERROR: No .txt files found in {DATA_DIR}/.")
        return 1

    print(f"Found {len(txt_files)} transcript file(s):")
    for f in txt_files:
        print(f" - {f.name}")
    print()

    # --- Phase 1: chunk all files ---------------------------------------------
    all_chunks: list[str] = []
    all_metas: list[dict] = []
    all_ids: list[str] = []

    for path in txt_files:
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_metadata(raw, path.name) 
        chunks = chunk_text(body)
        print(f" {path.name}: {len(body):,} chars {len(chunks)} chunks")

        for i, c in enumerate (chunks):
            all_chunks.append(c)
            all_metas.append({**meta, "chunk_index": i})
            all_ids.append(f"{path.stem}_chunk_{i:04d}")
    
    total_chunks = len(all_chunks)
    total_tokens = sum(len(TOKENIZER.encode(c)) for c in all_chunks)
    est_cost = (total_tokens / 1_000_000) * 0.02 # text-embedding-3-small price
    print()
    print(f"Total chunks: {total_chunks}")
    print(f"Total tokens to embed: {total_tokens:,}")
    print(f"Estimated cost: ${est_cost:.4f}")
    print()

    if args.dry_run:
        print("--dry-run set. Stopping before API calls.")
        return 0

    # --- Phase 2: connect to OpenAI + ChromaDB --------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI API KEY not set in env") 
        return 1
    client = OpenAI(api_key=api_key)

    chroma_client = chromadb.PersistentClient(
        path = str(CHROMA_DIR), 
        settings = Settings(anonymized_telemetry=False), 
    )

    if args.reset:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass # collection didn't exist

    collection = chroma_client.get_or_create_collection(
        name = COLLECTION_NAME, 
        metadata={"description": "Bank earnings call transcripts, Q1 2026"},
    )

    existing = collection.count()
    if existing and not args.reset:
        print(f"Collection already has {existing} chunks. Use --reset to re-ingest.")
        return 0

    # --- Phase 3: embed + store in batches ------------------------------------
    print(f"Embedding {total_chunks} chunks in batches of {EMBED_BATCH_SIZE}...")
    t0 = time.time()
    for batch_start in range(0, total_chunks, EMBED_BATCH_SIZE):
        batch_end = min(batch_start + EMBED_BATCH_SIZE, total_chunks) 
        batch_docs = all_chunks[batch_start:batch_end]
        batch_metas = all_metas[batch_start:batch_end]
        batch_ids = all_ids[batch_start:batch_end]
        
        embeddings = embed_batch(client, batch_docs) 
        collection.add( 
            ids=batch_ids, 
            documents=batch_docs, 
            embeddings=embeddings, 
            metadatas=batch_metas
        )
        print(f" [{batch_end}/{total_chunks}] stored")
    
    elapsed = time.time() - t0 
    print(f"\nDone. {total_chunks} chunks stored in {elapsed:.1f}s.") 
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} items.")

    # --- Phase 4: smoke-test retrieval ------------------------------------
    print("\n--- Smoke test: retrieving top 3 chunks for sample query ---") 
    test_q = "What did the CEO say about credit risk and loan loss provisions?" 
    print(f"Query: {test_q}\n") 
    q_emb = embed_batch(client, [test_q])[0] 
    results = collection.query(query_embeddings=[q_emb], n_results=3)

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):  
        print(f"Result {i+1} bank={meta['bank']} chunk={meta['chunk_index']} dist={dist:.3f}") 
        print(doc[:300].replace("\n", "") + "...") 
        print()
    
    return 0

if __name__ == "__main__": 
    sys.exit(main())