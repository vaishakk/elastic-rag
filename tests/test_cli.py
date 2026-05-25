from __future__ import annotations

from io import StringIO

from cli import format_result, print_results, prompt_search_loop
from rag.core.document import DocumentChunk


def test_format_result_includes_rank_ids_and_score():
    chunk = DocumentChunk(
        id="abc123",
        doc_id="doc-9",
        text="This is a sample chunk that should be wrapped into a readable block.",
        metadata={"score": 1.23456},
    )

    rendered = format_result(chunk, 1, width=40)

    assert "1. chunk_id=abc123 doc_id=doc-9 score=1.2346" in rendered
    assert "This is a sample chunk" in rendered


def test_print_results_shows_empty_state():
    buffer = StringIO()

    print_results("hello world", [], output=buffer)

    assert buffer.getvalue() == "\nQuery: hello world\nResults: 0\nNo results found.\n"


def test_prompt_search_loop_prompts_and_prints_results():
    class _FakeDB:
        def __init__(self):
            self.calls: list[str] = []

        def retrieve(self, query: str):
            self.calls.append(query)
            return [
                DocumentChunk(
                    id="chunk-1",
                    doc_id="doc-1",
                    text="Some useful result text.",
                    metadata={"score": 0.9},
                )
            ]

    inputs = iter(["search term", ""])
    output = StringIO()
    db = _FakeDB()

    def _input(prompt: str) -> str:
        assert prompt == "Search query (blank or 'quit' to exit): "
        return next(inputs)

    exit_code = prompt_search_loop(db, input_fn=_input, output=output)

    assert exit_code == 0
    assert db.calls == ["search term"]
    rendered = output.getvalue()
    assert "Interactive Elasticsearch RAG search" in rendered
    assert "Query: search term" in rendered
    assert "chunk_id=chunk-1 doc_id=doc-1 score=0.9000" in rendered
