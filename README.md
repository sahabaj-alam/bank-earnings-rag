# Bank Earnings RAG

A personal project that lets you ask natural-language questions across earnings call transcripts from major US banks (JPMorgan Chase, Goldman Sachs, Morgan Stanley, Bank of America, Wells Fargo).

**Status:** Week 2 / 4 - scaffolding (Jun 27, 2026)

## Why
Reading and comparing what bank executives say in earnings calls is slow. This app indexes public transcripts and lets you query them with citations.

## Stack
- Python 3.13
- OpenAI (gpt-4o-mini for chat, text-embedding-3-small for embeddings)
- ChromaDB (local vector store)
- Streamlit (UI + deployment)

## Disclaimer
This is a **personal project**, built on my own time, on my own equipment, using only **publicly available data** (SEC EDGAR filings, public investor-relations transcripts).

It is **not affiliated with, endorsed by, or related to my employer or any bank** whose data is referenced.

## Live demo
_Coming Week 3._

## Local setup (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env , paste your OpenAI API key
python src/hello_openai.py
```

## Screenshots

![Bank Earnings RAG UI](docs/screenshot_ui.png)

*Streamlit interfaces - ask a question, get a cited answers with source excerpts.*

## License
MIT - see [LICENSE](./LICENSE) for details.

## Data

This project ingests publicly available earnings call transcripts from SEC EDGAR and bank investor relations sites. Source attribution is preserved in each transcript file. Original content remains the property of its publishers; this project's MIT license applies only to the code, not the indexed transcripts.