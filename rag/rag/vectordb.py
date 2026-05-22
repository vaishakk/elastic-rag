from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from rag.core.document import Chunker, DocumentChunk, DocumentStack
from rag.core.embedding import EmbeddingModel
from rag.core.exceptions import VectorDBError
from rag.core.vectordb import VectorDB


DEFAULT_ES_URL = "https://localhost:9200"
DEFAULT_INDEX_NAME = "rag-documents"
DEFAULT_VECTOR_FIELD = "content_vector"
DEFAULT_CONTENT_FIELD = "content"
DEFAULT_K = 4
DEFAULT_NUM_CANDIDATES = 20
DEFAULT_SIMILARITY = "cosine"


class ElasticsearchVectorDB(VectorDB):
    """
    Elasticsearch-backed vector database using dense vectors and kNN search.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        chunker: Chunker,
        *,
        client: Elasticsearch | None = None,
        index_name: str | None = None,
        vector_field: str | None = None,
        content_field: str | None = None,
        top_k: int | None = None,
        num_candidates: int | None = None,
        dims: int | None = None,
        similarity: str | None = None,
    ):
        super().__init__(model=model)
        self.chunker = chunker
        self.client = client or self._build_client()
        self.index_name = index_name or os.environ.get("ES_INDEX_NAME", DEFAULT_INDEX_NAME)
        self.vector_field = vector_field or os.environ.get("ES_VECTOR_FIELD", DEFAULT_VECTOR_FIELD)
        self.content_field = content_field or os.environ.get("ES_CONTENT_FIELD", DEFAULT_CONTENT_FIELD)
        self.top_k = top_k if top_k is not None else self._env_int("ES_TOP_K", DEFAULT_K)
        self.num_candidates = (
            num_candidates if num_candidates is not None else self._env_int("ES_NUM_CANDIDATES", DEFAULT_NUM_CANDIDATES)
        )
        self.dims = dims
        self.similarity = similarity or os.environ.get("ES_SIMILARITY", DEFAULT_SIMILARITY)

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _build_client(self) -> Elasticsearch:
        es_url = os.environ.get("ES_URL", DEFAULT_ES_URL)
        password = os.environ.get("ELASTIC_PASSWORD")
        if not password:
            raise VectorDBError("ELASTIC_PASSWORD must be set for ElasticsearchVectorDB")

        ca_cert = os.environ.get("ES_CA_CERT")
        if ca_cert:
            ca_path = Path(ca_cert)
            if not ca_path.is_absolute():
                ca_path = self._project_root() / ca_path
        else:
            local_ca = self._project_root() / "http_ca.crt"
            if local_ca.exists():
                ca_path = local_ca
            else:
                es_home = os.environ.get("ES_HOME")
                if not es_home:
                    raise VectorDBError(
                        "Set ES_CA_CERT or ES_HOME, or place http_ca.crt in the project root"
                    )
                ca_path = Path(es_home) / "config/certs/http_ca.crt"

        if not ca_path.exists():
            raise VectorDBError(f"CA cert not found: {ca_path}")

        return Elasticsearch(
            es_url,
            basic_auth=("elastic", password),
            ca_certs=str(ca_path),
            request_timeout=30,
        )

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None or not value.strip():
            return default
        try:
            return int(value)
        except ValueError as exc:  # pragma: no cover - configuration error
            raise VectorDBError(f"Environment variable {name} must be an integer") from exc

    def _split(self, stack: DocumentStack) -> List[DocumentChunk]:
        try:
            return self.chunker.split_doc(stack)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise VectorDBError("Failed to chunk documents for Elasticsearch indexing") from exc

    def _embed_chunks(self, chunks: Sequence[DocumentChunk]):
        try:
            return self.model.embed_batch(list(chunks))
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise VectorDBError("Failed to embed chunks for Elasticsearch indexing") from exc

    def _infer_dims(self, vectors: Sequence[Sequence[float]]) -> int:
        if self.dims is not None:
            return self.dims
        if not vectors:
            raise VectorDBError("Cannot infer vector dimensions from an empty embedding set")
        return len(vectors[0])

    def _ensure_index(self, dims: int) -> None:
        if self.client.indices.exists(index=self.index_name):
            return

        mapping = {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                self.content_field: {"type": "text"},
                self.vector_field: {
                    "type": "dense_vector",
                    "dims": dims,
                    "similarity": self.similarity,
                    "index": True,
                },
            }
        }

        try:
            self.client.indices.create(index=self.index_name, mappings=mapping)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise VectorDBError(f"Failed to create Elasticsearch index {self.index_name}") from exc

    def _chunk_actions(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]):
        for chunk, vector in zip(chunks, vectors):
            yield {
                "_index": self.index_name,
                "_id": chunk.id,
                "_source": {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    self.content_field: chunk.text,
                    self.vector_field: list(vector),
                },
            }

    def _index_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return

        embeddings = self._embed_chunks(chunks)
        vectors = [embedding.embedding for embedding in embeddings]
        dims = self._infer_dims(vectors)
        self._ensure_index(dims)

        try:
            bulk(self.client, self._chunk_actions(chunks, vectors), refresh=True)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise VectorDBError("Failed to bulk index chunks into Elasticsearch") from exc

    def create_db(self, stack: DocumentStack):
        chunks = self._split(stack)
        if not chunks:
            raise VectorDBError("Document stack produced no chunks")

        self._index_chunks(chunks)

    def add_stack(self, stack: DocumentStack):
        chunks = self._split(stack)
        if not chunks:
            return
        self._index_chunks(chunks)

    def add_chunk(self, chunk: DocumentChunk):
        self._index_chunks([chunk])

    def retrieve(self, query: str, **kwargs) -> list[DocumentChunk]:
        top_k, num_candidates = self.top_k, self.num_candidates
        if 'top_k' in kwargs:
            top_k = kwargs.pop('top_k')
        if 'num_candidates' in kwargs:
            num_candidates = kwargs.pop('num_candidates')
        return self.search(query, top_k=top_k, num_candidates=num_candidates)

    def search(self, query: str, *, top_k: int | None = None, num_candidates: int | None = None) -> List[DocumentChunk]:
        query_chunk = DocumentChunk(id="__query__", doc_id="__query__", text=query)

        try:
            query_embedding = self.model.embed(query_chunk).embedding
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise VectorDBError("Failed to embed Elasticsearch search query") from exc

        k = top_k or self.top_k
        candidates = num_candidates or self.num_candidates
        if candidates < k:
            candidates = k

        knn_query = {
            "field": self.vector_field,
            "query_vector": query_embedding,
            "k": k,
            "num_candidates": candidates,
        }

        try:
            response = self.client.search(index=self.index_name, knn=knn_query, size=k)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise VectorDBError("Elasticsearch kNN search failed") from exc

        hits = response.get("hits", {}).get("hits", [])
        chunks: List[DocumentChunk] = []
        for hit in hits:
            source = hit.get("_source") or {}
            chunk_id = source.get("chunk_id") or hit.get("_id")
            doc_id = source.get("doc_id") or ""
            text = source.get(self.content_field) or ""
            chunks.append(DocumentChunk(id=chunk_id, doc_id=doc_id, text=text))

        return chunks
