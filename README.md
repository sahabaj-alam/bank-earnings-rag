# Bank Earnings RAG

A personal project that lets you ask natural-language questions across earnings call transcripts from major US banks (JPMC, Goldman Sachs, Morgan Stanley, BofA, Citi, Wells Fargo).

**Status:** Week 1 / 4 - scaffolding (Jun 14, 2026)

## Why
Reading and comparing what bank executives say in earnings calls is slow. This app indexes public transcripts and lets you query them with citations.

## Stack
- Python 3.13
- OpenAI (gpt-40-mini for chat, text-embedding-3-small for embeddings)
- ChromaDB (local vector store)
- Streamlit (UI + deployment)

## Disclaimer
This is a **personal project**, built on my own time, on my own equipment, using only **publicly available data** (SEC EDGAR filings, public investor-relations transcripts).

It is **not affiliated with, endorsed by, or related to my employer or any bank** whose data is referenced.

## Live demo
_Coming Week 3._

## Local setup (macOS)
```bash
python3 - venv venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env , paste your OpenAI API key
python src/hello_openai.py
```