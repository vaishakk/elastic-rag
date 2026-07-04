from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import api.api.main as main
from api import RAGAdapter, chunk2doc, doc2chunk
from api import Document, InferQuery
from rag import DocumentChunk


def test_build_rag_system_uses_supplied_index_name(monkeypatch):
    created: dict[str, object] = {}

    class FakeVectorDB:
        def __init__(self, model, chunker, index_name=None):
            created["vector_db"] = {
                "model": model,
                "chunker": chunker,
                "index_name": index_name,
            }

    class FakeRAG:
        def __init__(self, doc_stack, embed_model, db):
            created["rag"] = {
                "doc_stack": doc_stack,
                "embed_model": embed_model,
                "db": db,
            }

    monkeypatch.setattr(main, "DocumentStackFromMarkdownFolder", lambda folder, extractor: ("stack", folder, extractor))
    monkeypatch.setattr(main, "PyPDFExtractor", lambda: "extractor")
    monkeypatch.setattr(main, "OpenAIEmbeddingModel", lambda: "embed-model")
    monkeypatch.setattr(main, "LlamaIndexChunker", lambda: "chunker")
    monkeypatch.setattr(main, "ElasticsearchVectorDB", FakeVectorDB)
    monkeypatch.setattr(main, "RAG", FakeRAG)

    rag_system = main.build_rag_system(index_name="custom-index")

    assert isinstance(rag_system, FakeRAG)
    assert created["rag"]["doc_stack"] == ("stack", "./docs", "extractor")
    assert created["rag"]["embed_model"] == "embed-model"
    assert created["vector_db"] == {
        "model": "embed-model",
        "chunker": "chunker",
        "index_name": "custom-index",
    }


def test_root_route_returns_hello_world():
    client = TestClient(main.app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_search_route_uses_rag_system(monkeypatch):
    class FakeRAG:
        def retrieve(self, query, **kwargs):
            return [{"query": query, "kwargs": kwargs}]

    monkeypatch.setattr(main, "get_rag_system", lambda: FakeRAG())
    client = TestClient(main.app)

    response = client.get("/search/example")

    assert response.status_code == 200
    assert response.json() == {
        "answer": [{"query": "example", "kwargs": {}}],
        "q": "example",
    }


def test_rag_adapter_search_converts_chunks_to_documents():
    class FakeRAG:
        def retrieve(self, query, limit):
            self.query = query
            self.limit = limit
            return [
                DocumentChunk(id="c1", doc_id="doc-1", text="first"),
                DocumentChunk(id="c2", doc_id="doc-2", text="second"),
            ]

    adapter = RAGAdapter(FakeRAG())

    results = adapter.search("find this", limit=2)

    assert [doc.title for doc in results] == ["doc-1", "doc-2"]
    assert [doc.text for doc in results] == ["first", "second"]


def test_rag_adapter_infer_converts_context_and_returns_refs():
    class FakeRAG:
        def infer(self, query, context):
            self.query = query
            self.context = context
            return "answer text"

    context = [
        Document(title="one", text="alpha", url=Path("/tmp/one.pdf")),
        Document(title="two", text="beta"),
    ]
    adapter = RAGAdapter(FakeRAG())

    result = adapter.infer(InferQuery(query="what?", context=context))

    assert result.answer == "answer text"
    assert result.refs == [Path("/tmp/one.pdf")]
    assert adapter.rag_system.query == "what?"
    assert adapter.rag_system.context == [
        DocumentChunk(id="one.pdf", doc_id="one", text="alpha", metadata={"url": "/tmp/one.pdf"}),
        DocumentChunk(id="two", doc_id="two", text="beta", metadata=None),
    ]


def test_document_chunk_and_document_conversion_helpers():
    chunk = DocumentChunk(id="c1", doc_id="doc-1", text="hello")
    doc = chunk2doc(chunk)

    assert doc == Document(title="doc-1", text="hello")

    source = Document(title="report", text="payload", url=Path("/tmp/report.pdf"))
    converted = doc2chunk(source)

    assert converted == DocumentChunk(
        id="report.pdf",
        doc_id="report",
        text="payload",
        metadata={"url": "/tmp/report.pdf"},
    )
