from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag.core.document import Chunker, DocumentChunk, DocumentStack
from rag.core.exceptions import EmbeddingError, VectorDBError
from rag.rag.embedding import OpenAIEmbeddingModel
from rag.rag.vectordb import ElasticsearchVectorDB


@dataclass
class _FakeEmbeddingItem:
    embedding: list[float]


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingItem]


class _FakeEmbeddingsAPI:
    def __init__(self, fail_on_text: str | None = None, fail_times: int = 0, fail_message: str = "maximum context length exceeded"):
        self.calls: list[list[str]] = []
        self.fail_on_text = fail_on_text
        self.fail_times = fail_times
        self.fail_message = fail_message

    def create(self, model: str, input):
        texts = list(input)
        self.calls.append(texts)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError(self.fail_message)
        if self.fail_on_text and any(self.fail_on_text in text for text in texts):
            raise RuntimeError(self.fail_message)
        return _FakeEmbeddingResponse(
            data=[_FakeEmbeddingItem(embedding=[float(len(text))]) for text in texts]
        )


class _FakeOpenAIClient:
    def __init__(self, fail_on_text: str | None = None, fail_times: int = 0, fail_message: str = "maximum context length exceeded"):
        self.embeddings = _FakeEmbeddingsAPI(
            fail_on_text=fail_on_text,
            fail_times=fail_times,
            fail_message=fail_message,
        )


class _NoopChunker(Chunker):
    def split_doc(self, documents: DocumentStack):
        return []


def test_openai_embedding_model_batches_by_token_budget():
    client = _FakeOpenAIClient()
    model = OpenAIEmbeddingModel(
        client=client, batch_token_limit=19, batch_chunk_limit=10
    )

    chunks = [
        DocumentChunk(id="1", doc_id="d", text="a" * 30),
        DocumentChunk(id="2", doc_id="d", text="b" * 30),
        DocumentChunk(id="3", doc_id="d", text="c" * 30),
    ]

    embeddings = model.embed_batch(chunks)

    assert len(embeddings) == 3
    assert len(client.embeddings.calls) == 3
    assert client.embeddings.calls[0] == ["a" * 30]
    assert client.embeddings.calls[1] == ["b" * 30]
    assert client.embeddings.calls[2] == ["c" * 30]


def test_openai_embedding_model_splits_failed_large_batch():
    class _FailingEmbeddingsAPI(_FakeEmbeddingsAPI):
        def create(self, model: str, input):
            texts = list(input)
            self.calls.append(texts)
            if len(texts) > 1 and self.fail_on_text and any(self.fail_on_text in text for text in texts):
                raise RuntimeError("maximum context length exceeded")
            return _FakeEmbeddingResponse(
                data=[_FakeEmbeddingItem(embedding=[float(len(text))]) for text in texts]
            )

    class _FailingOpenAIClient:
        def __init__(self, fail_on_text: str | None = None):
            self.embeddings = _FailingEmbeddingsAPI(fail_on_text=fail_on_text)

    client = _FailingOpenAIClient(fail_on_text="x" * 30)
    model = OpenAIEmbeddingModel(
        client=client, batch_token_limit=10_000, batch_chunk_limit=10
    )

    chunks = [
        DocumentChunk(id="1", doc_id="d", text="short one"),
        DocumentChunk(id="2", doc_id="d", text="x" * 30),
        DocumentChunk(id="3", doc_id="d", text="short two"),
    ]

    embeddings = model.embed_batch(chunks)

    assert len(embeddings) == 3
    assert len(client.embeddings.calls) >= 2


def test_openai_embedding_model_rejects_single_chunk_over_budget():
    client = _FakeOpenAIClient()
    model = OpenAIEmbeddingModel(client=client, batch_token_limit=5, batch_chunk_limit=10)

    chunk = DocumentChunk(id="1", doc_id="d", text="a" * 100)

    with pytest.raises(EmbeddingError, match="exceeds the configured embedding batch token limit"):
        model.embed_batch([chunk])


def test_openai_embedding_model_retries_tpm_errors(monkeypatch):
    client = _FakeOpenAIClient(fail_times=2, fail_message="tokens per minute limit exceeded")
    model = OpenAIEmbeddingModel(
        client=client,
        batch_token_limit=10_000,
        batch_chunk_limit=10,
        retry_max_attempts=4,
        retry_base_delay_seconds=0.01,
    )

    sleeps: list[float] = []
    monkeypatch.setattr("rag.rag.embedding.time.sleep", lambda delay: sleeps.append(delay))

    chunks = [
        DocumentChunk(id="1", doc_id="d", text="short one"),
        DocumentChunk(id="2", doc_id="d", text="short two"),
    ]

    embeddings = model.embed_batch(chunks)

    assert len(embeddings) == 2
    assert len(client.embeddings.calls) == 3
    assert sleeps == [0.01, 0.02]


def test_vector_db_uses_model_embed_batch():
    class _Model:
        def __init__(self):
            self.calls = 0

        def embed_batch(self, chunks):
            self.calls += 1
            return [type("E", (), {"embedding": [1.0]})() for _ in chunks]

    class _Client:
        class indices:
            @staticmethod
            def exists(index):
                return True

        @staticmethod
        def search(index, knn, size):
            return {"hits": {"hits": []}}

    class _DB(ElasticsearchVectorDB):
        def _build_client(self):
            return _Client()

    model = _Model()
    db = _DB(model=model, chunker=_NoopChunker(), client=_Client())

    result = db._embed_chunks(
        [
            DocumentChunk(id="1", doc_id="d", text="one"),
            DocumentChunk(id="2", doc_id="d", text="two"),
        ]
    )

    assert model.calls == 1
    assert len(result) == 2


def test_vector_db_skips_chunks_that_already_exist(monkeypatch):
    indexed_actions: list[dict] = []

    class _Model:
        def __init__(self):
            self.calls = 0

        def embed_batch(self, chunks):
            self.calls += 1
            return [type("E", (), {"embedding": [float(len(chunk.text))]})() for chunk in chunks]

    class _Client:
        def __init__(self):
            self.mget_calls: list[tuple[str, list[str]]] = []

        class indices:
            @staticmethod
            def exists(index):
                return True

        def mget(self, index, ids):
            self.mget_calls.append((index, ids))
            return {
                "docs": [
                    {"_id": "existing", "found": True},
                    {"_id": "missing", "found": False},
                ]
            }

    class _DB(ElasticsearchVectorDB):
        def _build_client(self):
            return _Client()

    def _fake_bulk(client, actions, refresh):
        indexed_actions.extend(list(actions))
        return len(indexed_actions), []

    monkeypatch.setattr("rag.rag.vectordb.bulk", _fake_bulk)

    model = _Model()
    client = _Client()
    db = _DB(model=model, chunker=_NoopChunker(), client=client)

    db._index_chunks(
        [
            DocumentChunk(id="existing", doc_id="d", text="already indexed"),
            DocumentChunk(id="new", doc_id="d", text="should be indexed"),
            DocumentChunk(id="existing", doc_id="d", text="duplicate existing"),
        ]
    )

    assert model.calls == 1
    assert client.mget_calls == [("rag-documents", ["existing", "new"])]
    assert [action["_id"] for action in indexed_actions] == ["new"]


def test_vector_db_retrieve_selects_bm25_search():
    class _Model:
        def embed(self, chunk):
            raise AssertionError("vector search should not be used")

    class _Client:
        class indices:
            @staticmethod
            def exists(index):
                return True

        def __init__(self):
            self.calls = []

        def search(self, index, **kwargs):
            self.calls.append((index, kwargs))
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "c1",
                            "_score": 1.23,
                            "_source": {
                                "chunk_id": "c1",
                                "doc_id": "d1",
                                "content": "matched text",
                            },
                        }
                    ]
                }
            }

    class _DB(ElasticsearchVectorDB):
        def _build_client(self):
            return _Client()

    client = _Client()
    db = _DB(model=_Model(), chunker=_NoopChunker(), client=client)

    results = db.retrieve("diet and cancer", search_method="bm25", top_k=5)

    assert len(results) == 1
    assert results[0].id == "c1"
    assert results[0].doc_id == "d1"
    assert results[0].text == "matched text"
    assert results[0].metadata == {"score": 1.23}
    assert client.calls == [
        (
            "rag-documents",
            {
                "body": {
                    "query": {
                        "match": {
                            "content": {
                                "query": "diet and cancer",
                            }
                        }
                    },
                    "size": 5,
                },
            },
        )
    ]


def test_vector_db_retrieve_defaults_to_vector_search():
    class _Model:
        def __init__(self):
            self.calls = 0

        def embed(self, chunk):
            self.calls += 1
            return type("E", (), {"embedding": [1.0, 2.0]})()

    class _Client:
        class indices:
            @staticmethod
            def exists(index):
                return True

        def __init__(self):
            self.calls = []

        def search(self, index, **kwargs):
            self.calls.append((index, kwargs))
            return {"hits": {"hits": []}}

    class _DB(ElasticsearchVectorDB):
        def _build_client(self):
            return _Client()

    client = _Client()
    model = _Model()
    db = _DB(model=model, chunker=_NoopChunker(), client=client)

    results = db.retrieve("diet and cancer", top_k=3, num_candidates=7)

    assert results == []
    assert model.calls == 1
    assert client.calls == [
        (
            "rag-documents",
            {
                "knn": {
                    "field": "content_vector",
                    "query_vector": [1.0, 2.0],
                    "k": 3,
                    "num_candidates": 7,
                },
                "size": 3,
            },
        )
    ]


def test_vector_db_retrieve_hybrid_search_fuses_vector_and_bm25_results():
    class _Model:
        def __init__(self):
            self.calls = 0

        def embed(self, chunk):
            self.calls += 1
            return type("E", (), {"embedding": [1.0, 2.0]})()

    class _Client:
        class indices:
            @staticmethod
            def exists(index):
                return True

        def __init__(self):
            self.calls = []

        def search(self, index, **kwargs):
            self.calls.append((index, kwargs))
            if "knn" in kwargs:
                return {
                    "hits": {
                        "hits": [
                            {
                                "_id": "c1",
                                "_score": 9.0,
                                "_source": {
                                    "chunk_id": "c1",
                                    "doc_id": "d1",
                                    "content": "vector first",
                                },
                            },
                            {
                                "_id": "c2",
                                "_score": 8.0,
                                "_source": {
                                    "chunk_id": "c2",
                                    "doc_id": "d2",
                                    "content": "vector second",
                                },
                            },
                        ]
                    }
                }
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "c2",
                            "_score": 7.0,
                            "_source": {
                                "chunk_id": "c2",
                                "doc_id": "d2",
                                "content": "bm25 second",
                            },
                        },
                        {
                            "_id": "c3",
                            "_score": 6.0,
                            "_source": {
                                "chunk_id": "c3",
                                "doc_id": "d3",
                                "content": "bm25 third",
                            },
                        },
                    ]
                }
            }

    class _DB(ElasticsearchVectorDB):
        def _build_client(self):
            return _Client()

    client = _Client()
    model = _Model()
    db = _DB(model=model, chunker=_NoopChunker(), client=client)

    results = db.retrieve("diet and cancer", top_k=2, num_candidates=7, search_method="hybrid")

    assert [chunk.id for chunk in results] == ["c2", "c1", "c3"]
    assert results[0].metadata["vector_score"] == 8.0
    assert results[0].metadata["bm25_score"] == 7.0
    assert results[0].metadata["hybrid_score"] > results[1].metadata["hybrid_score"]
    assert results[0].metadata["score"] == results[0].metadata["hybrid_score"]
    assert results[1].metadata["vector_score"] == 9.0
    assert "bm25_score" not in results[1].metadata
    assert results[2].metadata["bm25_score"] == 6.0
    assert len(client.calls) == 2
    assert "knn" in client.calls[0][1]
    assert "body" in client.calls[1][1]


def test_vector_db_retrieve_rejects_unknown_search_method():
    class _Model:
        def embed(self, chunk):
            return type("E", (), {"embedding": [1.0, 2.0]})()

    class _Client:
        class indices:
            @staticmethod
            def exists(index):
                return True

    class _DB(ElasticsearchVectorDB):
        def _build_client(self):
            return _Client()

    db = _DB(model=_Model(), chunker=_NoopChunker(), client=_Client())

    with pytest.raises(VectorDBError, match="Unsupported search method"):
        db.retrieve("diet and cancer", search_method="unknown")
