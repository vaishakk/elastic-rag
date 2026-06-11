# Elastic RAG

An easy-to-use **Retrieval-Augmented Generation (RAG)** pipeline built on **Elasticsearch** with pluggable components for chunking, embeddings, retrieval, and evaluation.

The goal of this repository is to provide a small, practical RAG stack that is straightforward to run, extend, and reuse in other projects. The design keeps the core pieces modular so different adapters, chunkers, embedders, vector stores, and inference components can be swapped in over time.

This project is useful for experimenting with:

- Elasticsearch-backed RAG
- vector search
- semantic search
- document chunking
- embedding pipelines
- retrieval evaluation
- precision and recall measurement
- modular Python AI application architecture

## Status

This project is still a work in progress.

Current support is intentionally narrow:

- chunking is based on basic LlamaIndex chunking
- embeddings are currently provided through OpenAI
- Elasticsearch is used for storing and searching documents
- retrieval supports vector, BM25, and hybrid BM25 + vector search
- the main interface is an interactive search CLI
- retrieval evaluation is supported through precision/recall utilities

The longer-term plan is to support a wider range of tools and integrations while keeping the system simple to compose.

## Why This Project Exists

Many RAG demos focus only on sending retrieved chunks to an LLM. In real-world systems, the harder engineering problems often appear earlier in the pipeline:

- how documents are chunked
- how metadata is preserved
- how embeddings are generated and stored
- how retrieval quality is measured
- how search components can be replaced without rewriting the whole system
- how to keep the codebase small, inspectable, and extensible

Elastic RAG focuses on these fundamentals.

It is designed as a practical foundation for building and evaluating RAG systems with Elasticsearch.

## Design

- `rag/core` contains shared domain types, interfaces, and evaluation helpers.
- `rag/adapters` contains input and output adapters.
- `rag/rag` contains concrete implementations for the current stack.
- `main.py` expose a small interactive search CLI.

## Project Shape

- the core abstractions are explicit and small
- the current implementation is easy to inspect and evolve incrementally
- new adapters and providers should fit into the existing interfaces without forcing a rewrite
- retrieval evaluation is treated as a first-class part of the system rather than an afterthought

If you are evaluating the project for reuse, expect the current surface area to be minimal, stable where possible, and open to extension.

## Architecture

```text
Input documents
      │
      ▼
Text chunking
      │
      ▼
Embedding generation
      │
      ▼
Elasticsearch indexing
      │
      ▼
Vector / semantic retrieval
      │
      ▼
CLI search interface + evaluation
```

## Setup

1. Install dependencies.

   ```bash
   uv sync
   ```

2. Set up an Elasticsearch instance.

   The project requires Elasticsearch to be available before the CLI or indexing workflow can run.

3. Create a `.env` file in the project root.

   The repository does not include one, so this file needs to be created locally.

   ```bash
   touch .env
   ```

4. Add the required environment variables to `.env`.

   Required:

   ```bash
   ELASTIC_PASSWORD=your_elasticsearch_password
   OPENAI_API_KEY=your_openai_api_key
   ```

   Optional overrides:

   ```bash
   ES_URL=https://localhost:9200
   ES_CA_CERT=/path/to/http_ca.crt
   ES_INDEX_NAME=test-documents
   ES_CONTENT_FIELD=content
   ES_VECTOR_FIELD=embedding
   ES_SIMILARITY=cosine
   ES_HOME=/path/to/elasticsearch
   ```

### Environment Variables

| Variable | Required | Description |
|---|---:|---|
| `ELASTIC_PASSWORD` | Yes | Password for the Elasticsearch instance. |
| `OPENAI_API_KEY` | Yes | API key used for generating embeddings. |
| `ES_URL` | No | Elasticsearch URL if your cluster is not running on the default host. |
| `ES_CA_CERT` | No | Custom CA certificate path. |
| `ES_INDEX_NAME` | No | Override for the default Elasticsearch index. |
| `ES_CONTENT_FIELD` | No | Override for the stored text field. |
| `ES_VECTOR_FIELD` | No | Override for the stored embedding field. |
| `ES_SIMILARITY` | No | Override for vector similarity setting. |
| `ES_HOME` | No | Used only if relying on Elasticsearch's bundled CA certificate path and `ES_CA_CERT` is not set. |

## How To Run

1. Make sure Elasticsearch is running.

2. Ensure `.env` exists in the repository root and contains the variables above.

3. Start the interactive CLI.

   ```bash
   uv run cli.py
   ```

4. Use the menu to either search the current index or reindex all PDF files under `./docs`.

5. If you choose search, enter a query when prompted.

6. If you choose reindex, the CLI will rebuild the index from the recursive contents of `./docs`.

7. Press Enter on a blank line or type `quit` to exit.

## API

The project also exposes a small FastAPI app.

Start it with:

```bash
uv run uvicorn api.api.main:app --reload
```

Available routes:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health-style root response. |
| `GET` | `/search/{query}` | Runs retrieval for the supplied query string and returns the answer payload. |

Example:

```bash
curl http://127.0.0.1:8000/search/example
```

Example response:

```json
{
  "answer": [],
  "q": "example"
}
```

## Retrieval Evaluation

This project includes utilities for measuring retrieval quality using precision and recall.

The evaluation workflow is intended to support questions such as:

- are the retrieved chunks relevant?
- how does retrieval quality change with different chunking strategies?
- what happens when top-k is changed?
- how should retrieval be improved before adding more complex generation logic?

Evaluation-oriented RAG development is especially important because a language model can only produce grounded answers if the retriever provides useful context.

The current benchmark on `rag/evals/test.tsv` and `rag/evals/queries.jsonl` is:

| Search Method | Top k | Precision | Recall | F-Score |
|---------------|---:|---:|---:|---:|
| vector        | 10 | 0.29 | 0.18 | 0.22 |
| vector        | 4 | 0.38 | 0.12 | 0.18 |
| bm25          | 10 | 0.31 | 0.17 | 0.22 |
| bm25          | 4 | 0.37 | 0.12 | 0.18 |
| hybrid        | 10 | 0.27 | 0.19 | 0.22 |
| hybrid        | 4 | 0.37 | 0.13 | 0.19 |

## Roadmap

Planned improvements include:

- richer document ingestion
- improved FastAPI service layer
- Docker Compose setup
- citation-aware answer generation
- local embedding model support
- evaluation reports
- GitHub Actions CI
- example datasets and query transcripts

## Welcoming Collaborations

Contributions, ideas, and discussions are welcome.

I am especially interested in collaborating on:

- Elasticsearch-based RAG systems
- retrieval evaluation and benchmarking
- hybrid retrieval tuning
- chunking strategies for noisy documents
- citation-aware answer generation
- lightweight RAG APIs and developer tooling
- production-oriented LLM application architecture

Good first areas for contribution include:

- adding a Docker Compose setup
- improving setup documentation
- adding example datasets
- adding tests around retrieval behavior
- improving evaluation scripts
- adding support for more embedding providers

If you are working on search, RAG, LLM applications, information retrieval, or AI engineering, feel free to open an issue, suggest an improvement, or connect with me.

## Notes

- Python 3.12 or newer is required.
- Elasticsearch is used for storing and searching documents.
- The interactive CLI prompts for a search query and prints the top results.
- The project is intentionally small and inspectable so the architecture is easy to understand.

## About

Elastic RAG is a practical Elasticsearch-backed RAG pipeline focused on modular design, semantic search, and retrieval evaluation.

It is built as a learning, experimentation, and portfolio project for AI Engineering, GenAI Engineering, RAG Engineering, and LLM application development.
