# Elastic RAG

Modular RAG pipeline with Elasticsearch used for document storage and search.

## Setup

1. Install dependencies.
   ```bash
   uv sync
   ```
2. Set the required environment variables in `.env`.
   - `ELASTIC_PASSWORD`
   - `ES_URL` if your cluster is not running on the default host
   - `ES_CA_CERT` if you want to override the bundled CA certificate path

## Architecture

The RAG architecture lives in the `rag/` package.

- `rag/core` contains shared domain types and interfaces.
- `rag/adapters` contains adapters for external inputs and outputs.
- `rag/rag` contains concrete RAG components and implementations.
- `rag/main.py` is the package entrypoint.

## Notes

- Python 3.12 or newer is required.
- Elasticsearch is used for storing and searching documents.
