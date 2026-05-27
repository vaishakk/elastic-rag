from __future__ import annotations

from io import StringIO
from pathlib import Path

from cli import format_result, print_results, prompt_main_menu, prompt_search_loop, reindex_docs_folder
from rag.core.document import Document, DocumentChunk, DocumentStack


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


def test_reindex_docs_folder_rebuilds_index():
    class _FakeIndices:
        def __init__(self):
            self.deleted: list[str] = []

        def exists(self, index):
            return True

        def delete(self, index):
            self.deleted.append(index)

    class _FakeClient:
        def __init__(self):
            self.indices = _FakeIndices()

    class _FakeDB:
        def __init__(self):
            self.client = _FakeClient()
            self.index_name = "test-documents"
            self.created_with = None

        def create_db(self, stack):
            self.created_with = stack

    def _stack_factory(folder_url: str, extractor):
        assert Path(folder_url) == Path("./docs")
        assert extractor.__class__.__name__ == "PyPDFExtractor"
        return DocumentStack(
            documents=[
                Document(id="1", path=Path("docs/a.pdf"), title="A", text="one"),
                Document(id="2", path=Path("docs/b.pdf"), title="B", text="two"),
            ]
        )

    db = _FakeDB()
    buffer = StringIO()

    reindex_docs_folder(
        db,
        output=buffer,
        stack_factory=_stack_factory,
        extractor_factory=lambda: type("PyPDFExtractor", (), {})(),
    )

    assert db.client.indices.deleted == ["test-documents"]
    assert db.created_with is not None
    assert len(db.created_with.documents) == 2
    assert "Reindexed 2 document(s) from docs" in buffer.getvalue()


def test_prompt_main_menu_routes_to_search_then_quit():
    class _FakeDB:
        def __init__(self):
            self.calls: list[str] = []
            self.client = type("Client", (), {"indices": type("Indices", (), {"exists": staticmethod(lambda index: False), "delete": staticmethod(lambda index: None)})()})()
            self.index_name = "test-documents"

        def retrieve(self, query: str):
            self.calls.append(query)
            return []

        def create_db(self, stack):
            raise AssertionError("not expected")

    inputs = iter(["1", "search term", "", "q"])
    output = StringIO()
    db = _FakeDB()

    def _input(prompt: str) -> str:
        return next(inputs)

    exit_code = prompt_main_menu(db, input_fn=_input, output=output)

    assert exit_code == 0
    assert db.calls == ["search term"]
    rendered = output.getvalue()
    assert "Interactive Elasticsearch RAG" in rendered
    assert "No results found." in rendered
