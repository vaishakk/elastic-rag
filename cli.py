from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Callable, Iterable, TextIO

from rag import DocumentChunk, DocumentStackFromPDFFolder, ElasticsearchVectorDB, LlamaIndexChunker, OpenAIEmbeddingModel
from rag.rag.pdf_document_reader import PyPDFExtractor


DEFAULT_INDEX_NAME = "test-documents"
DEFAULT_DOCS_DIR = Path("./docs")
DEFAULT_MENU_PROMPT = "Select [1] search, [2] reindex ./docs, [q] quit: "
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


def reindex_docs_folder(
    db: ElasticsearchVectorDB,
    docs_dir: Path = DEFAULT_DOCS_DIR,
    *,
    output: TextIO = sys.stdout,
    stack_factory: Callable[[str, PyPDFExtractor], DocumentStackFromPDFFolder] = DocumentStackFromPDFFolder,
    extractor_factory: Callable[[], PyPDFExtractor] = PyPDFExtractor,
) -> None:
    stack = stack_factory(str(docs_dir), extractor_factory())

    if db.client.indices.exists(index=db.index_name):
        db.client.indices.delete(index=db.index_name)

    db.create_db(stack)
    print(f"Reindexed {len(stack.documents)} document(s) from {docs_dir}", file=output)


def prompt_search_loop(
    db: ElasticsearchVectorDB,
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    prompt: str = DEFAULT_PROMPT,
    exit_message: str | None = "Exiting.",
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
            if exit_message:
                print(exit_message, file=output)
            return 0

        try:
            results = db.retrieve(query)
        except Exception as exc:  # pragma: no cover - runtime environment/config errors
            print(f"Search failed: {exc}", file=output)
            continue

        print_results(query, results, output=output)


def prompt_main_menu(
    db: ElasticsearchVectorDB,
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> int:
    while True:
        print("\nInteractive Elasticsearch RAG", file=output)
        print("1. Search current index", file=output)
        print("2. Reindex all docs in ./docs", file=output)
        print("q. Quit", file=output)

        try:
            choice = input_fn(DEFAULT_MENU_PROMPT).strip().lower()
        except EOFError:
            print(file=output)
            return 0

        if choice in {"q", "quit", "exit"}:
            print("Exiting.", file=output)
            return 0
        if choice in {"1", "s", "search"}:
            prompt_search_loop(db, input_fn=input_fn, output=output, exit_message=None)
            continue
        if choice in {"2", "r", "reindex"}:
            try:
                reindex_docs_folder(db, output=output)
            except Exception as exc:  # pragma: no cover - runtime environment/config errors
                print(f"Reindex failed: {exc}", file=output)
            continue

        print("Unknown choice. Enter 1, 2, or q.", file=output)


def main(argv: list[str] | None = None) -> int:
    _ = argv  # The CLI is intentionally interactive, so argv is unused.
    db = build_search_db()
    return prompt_main_menu(db)

if __name__ == "__main__":
    raise SystemExit(main())
