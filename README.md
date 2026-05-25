# Elastic RAG

An easy-to-use RAG pipeline built on Elasticsearch with pluggable components.

The goal of this repository is to provide a small, practical retrieval-augmented generation stack that is straightforward to run, extend, and reuse in other projects. The design keeps the core pieces modular so different adapters, chunkers, embedders, and inference components can be swapped in over time.

## Status

This project is still a work in progress.

Current support is intentionally narrow:
- chunking is based on basic LlamaIndex chunking
- embeddings are currently provided through OpenAI

The longer-term plan is to support a much wider range of tools and integrations while keeping the system simple to compose.

## Design

- `rag/core` contains shared domain types, interfaces, and evaluation helpers.
- `rag/adapters` contains input and output adapters.
- `rag/rag` contains concrete implementations for the current stack.
- `main.py` and `rag/main.py` expose a small interactive search CLI.

## Project Shape

- the core abstractions are explicit and small
- the current implementation is easy to inspect and evolve incrementally
- new adapters and providers should fit into the existing interfaces without forcing a rewrite

If you are evaluating the project for reuse, expect the current surface area to be minimal, stable where possible, and open to extension.

## Setup

1. Install dependencies.
   ```bash
   uv sync
   ```
2. Set up an Elasticsearch instance. The project requires Elasticsearch to be available before the CLI or indexing workflow can run.
3. Create a `.env` file in the project root. The repository does not include one, so this file needs to be created locally.
4. Add the required environment variables to `.env`.
   - `ELASTIC_PASSWORD`
   - `OPENAI_API_KEY`
   - `ES_URL` if your cluster is not running on the default host
   - `ES_CA_CERT` if you want to override the bundled CA certificate path
   - `ES_INDEX_NAME` if you want to override the default Elasticsearch index
   - `ES_CONTENT_FIELD` if you want to override the stored text field
   - `ES_VECTOR_FIELD` if you want to override the stored embedding field
   - `ES_SIMILARITY` if you want to override the vector similarity setting
   - `ES_HOME` only if you rely on Elasticsearch's bundled CA certificate path and do not set `ES_CA_CERT`

## How To Run

1. Make sure Elasticsearch is running and the target index is available.
2. Ensure `.env` exists in the repository root and contains the variables above.
3. Start the interactive CLI.
   ```bash
   uv run main.py
   ```
4. Enter a search query when prompted.
5. Press Enter on a blank line or type `quit` to exit.

## Notes

- Python 3.12 or newer is required.
- Elasticsearch is used for storing and searching documents.
- The interactive CLI prompts for a search query and prints the top results.
