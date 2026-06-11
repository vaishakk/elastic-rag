from __future__ import annotations

from pathlib import Path

import index_corpus


def test_index_corpus_creates_index_and_indexes_documents(tmp_path: Path, monkeypatch):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                '{"_id": "d1", "title": "First", "text": "alpha", "metadata": {"url": "https://example.com/a"}}',
                '{"_id": "d2", "title": "Second", "text": "beta"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    created = {}

    class FakeEmbeddingModel:
        pass

    class FakeChunker:
        pass

    class FakeStack:
        def __init__(self, file_url):
            created["stack_file_url"] = file_url
            self.documents = [
                type("Document", (), {"id": "d1", "title": "First", "text": "alpha"})(),
                type("Document", (), {"id": "d2", "title": "Second", "text": "beta"})(),
            ]

    class FakeDB:
        def __init__(self, model, chunker, client, index_name):
            created["init"] = {
                "model": model,
                "chunker": chunker,
                "client": client,
                "index_name": index_name,
            }

        def create_db(self, stack):
            created["stack"] = stack

    monkeypatch.setattr(index_corpus, "OpenAIEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(index_corpus, "LlamaIndexChunker", FakeChunker)
    monkeypatch.setattr(index_corpus, "ElasticsearchVectorDB", FakeDB)
    monkeypatch.setattr(index_corpus, "DocumentStackFromJSONLFile", FakeStack)

    indexed = index_corpus.index_corpus(
        object(),
        corpus_path,
        "nfcorp-documents",
    )

    assert indexed == 2
    assert created["stack_file_url"] == corpus_path
    assert created["init"]["index_name"] == "nfcorp-documents"
    assert isinstance(created["init"]["model"], FakeEmbeddingModel)
    assert isinstance(created["init"]["chunker"], FakeChunker)

    stack = created["stack"]
    assert [doc.id for doc in stack.documents] == ["d1", "d2"]
    assert [doc.title for doc in stack.documents] == ["First", "Second"]
    assert [doc.text for doc in stack.documents] == ["alpha", "beta"]
