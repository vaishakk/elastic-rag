from __future__ import annotations

import os
import sys
import textwrap
from typing import Callable, Iterable, TextIO

from rag import DocumentChunk, ElasticsearchVectorDB, LlamaIndexChunker, OpenAIEmbeddingModel


DEFAULT_INDEX_NAME = "test-documents"
DEFAULT_PROMPT = "Search query (blank or 'quit' to exit): "


def build_search_db(index_name: str | None = None) -> ElasticsearchVectorDB:
    embed_model = OpenAIEmbeddingModel()
    chunker = LlamaIndexChunker()
    resolved_index_name = index_name or os.environ.get("ES_INDEX_NAME", DEFAULT_INDEX_NAME)
    return ElasticsearchVectorDB(model=embed_model, chunker=chunker, index_name=resolved_index_name)


def format_result(chunk: DocumentChunk, rank: int, *, width: int = 88) -> str:
    score = None
    if chunk.metadata and "score" in chunk.metadata:
        score = chunk.metadata["score"]

    header = f"{rank}. chunk_id={chunk.id} doc_id={chunk.doc_id}"
    if score is not None:
        header += f" score={float(score):.4f}"

    text = chunk.text.strip() or "(empty chunk)"
    wrapped_text = textwrap.fill(text, width=width, subsequent_indent="    ")

    return f"{header}\n    {wrapped_text}"


def print_results(query: str, results: Iterable[DocumentChunk], *, output: TextIO = sys.stdout) -> None:
    results = list(results)
    print(f"\nQuery: {query}", file=output)
    print(f"Results: {len(results)}", file=output)
    if not results:
        print("No results found.", file=output)
        return

    for rank, chunk in enumerate(results, start=1):
        print(format_result(chunk, rank), file=output)
        if rank != len(results):
            print(file=output)


def prompt_search_loop(
    db: ElasticsearchVectorDB,
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    prompt: str = DEFAULT_PROMPT,
) -> int:
    print("Interactive Elasticsearch RAG search", file=output)
    print("Press Enter on an empty line or type 'quit' to exit.", file=output)

    while True:
        try:
            query = input_fn(prompt).strip()
        except EOFError:
            print(file=output)
            return 0

        if not query or query.lower() in {"quit", "exit"}:
            print("Exiting.", file=output)
            return 0

        try:
            results = db.retrieve(query)
        except Exception as exc:  # pragma: no cover - runtime environment/config errors
            print(f"Search failed: {exc}", file=output)
            continue

        print_results(query, results, output=output)


def main(argv: list[str] | None = None) -> int:
    _ = argv  # The CLI is intentionally interactive, so argv is unused.
    db = build_search_db()
    return prompt_search_loop(db)

if __name__ == "__main__":
    raise SystemExit(main())