# NLP_minutes

Meeting-minutes ingestion and retrieval system (PDF -> structured summaries -> vector search).

## What was added

This repo now includes two CLI scripts so you can use the system without the browser UI:

- `ingest_minutes.py` — ingest a PDF, parse/summarize minutes, and save to ChromaDB
- `query_minutes.py` — query minutes from CLI and optionally generate an LLM answer

These reuse the existing logic in `app.py` (header extraction, minute parsing, summarization, embedding, and Chroma storage).

## Setup

1. Create environment (example):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or install equivalent deps from environment.yml
```

2. Ensure `.env` exists with your API key(s), e.g. `OPENAI_API_KEY`.

3. Ensure OCR/PDF tools are installed on macOS if needed by `unstructured`:
- `tesseract`
- `poppler` (for `pdfinfo`)

## CLI usage

### Ingest a minutes PDF

```bash
python ingest_minutes.py "/path/to/minutes.pdf"
```

Preview only (no DB write):

```bash
python ingest_minutes.py "/path/to/minutes.pdf" --dry-run
```

### Query minutes

```bash
python query_minutes.py "What was decided about workload KPI?"
```

Top-k + retrieval only (no final LLM synthesis):

```bash
python query_minutes.py "AWS migration timeline" --top-k 5 --no-llm
```

## Notes

- `uploads/` stores working copies of PDFs used for parsing.
- `chroma_storage/` is persistent local vector storage.
- Keep `.env` private (never commit secrets).
