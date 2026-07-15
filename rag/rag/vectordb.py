from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sqlalchemy.testing.suite.test_reflection import metadata

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
DEFAULT_SEARCH_METHOD = "vector"
DEFAULT_RRF_K = 60
DEFAULT_SEARCH_METHOD = "vector"

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
        skip_indexing: bool = False,
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
        self.skip_indexing = skip_indexing

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

    @staticmethod
    def _dedupe_chunks(chunks: Sequence[DocumentChunk]) -> List[DocumentChunk]:
        deduped: List[DocumentChunk] = []
        seen_ids: set[str] = set()
        for chunk in chunks:
            if chunk.id in seen_ids:
                continue
            seen_ids.add(chunk.id)
            deduped.append(chunk)
        return deduped

    def _existing_chunk_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        unique_ids = list(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return set()
        if not self.client.indices.exists(index=self.index_name):
            return set()

        try:
            response = self.client.mget(index=self.index_name, ids=unique_ids)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise VectorDBError("Failed to check existing chunk ids in Elasticsearch") from exc

        docs = response.get("docs", []) if isinstance(response, dict) else []
        existing_ids: set[str] = set()
        for doc in docs:
            if doc.get("found"):
                doc_id = doc.get("_id")
                if doc_id:
                    existing_ids.add(doc_id)
        return existing_ids

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

        chunks = self._dedupe_chunks(chunks)
        if not chunks:
            return

        existing_ids = self._existing_chunk_ids([chunk.id for chunk in chunks])
        chunks = [chunk for chunk in chunks if chunk.id not in existing_ids]
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

    def _hits_to_chunks(self, hits) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for hit in hits:
            source = hit.get("_source") or {}
            chunk_id = source.get("chunk_id") or hit.get("_id")
            doc_id = source.get("doc_id") or ""
            text = source.get(self.content_field) or ""
            metadata = {
                "score": hit.get("_score") or 0,
            }
            chunks.append(DocumentChunk(id=chunk_id, doc_id=doc_id, text=text, metadata=metadata))
        return chunks

    @staticmethod
    def _rrf_score(rank: int, k: int = DEFAULT_RRF_K) -> float:
        return 1.0 / (k + rank)

    def _merge_search_results(
        self,
        vector_hits: List[DocumentChunk],
        bm25_hits: List[DocumentChunk],
        *,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> List[DocumentChunk]:
        merged: dict[str, DocumentChunk] = {}
        vector_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}
        vector_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            vector_ranks[hit.id] = rank
            vector_scores[hit.id] = float(hit.metadata.get("score", 0.0) if hit.metadata else 0.0)
            merged.setdefault(hit.id, hit)

        for rank, hit in enumerate(bm25_hits, start=1):
            bm25_ranks[hit.id] = rank
            bm25_scores[hit.id] = float(hit.metadata.get("score", 0.0) if hit.metadata else 0.0)
            merged.setdefault(hit.id, hit)

        ranked: list[tuple[float, str]] = []
        for chunk_id, chunk in merged.items():
            vector_rank = vector_ranks.get(chunk_id)
            bm25_rank = bm25_ranks.get(chunk_id)
            hybrid_score = 0.0
            if vector_rank is not None:
                hybrid_score += self._rrf_score(vector_rank, k=rrf_k)
            if bm25_rank is not None:
                hybrid_score += self._rrf_score(bm25_rank, k=rrf_k)

            metadata = dict(chunk.metadata or {})
            if chunk_id in vector_scores:
                metadata["vector_score"] = vector_scores[chunk_id]
            if chunk_id in bm25_scores:
                metadata["bm25_score"] = bm25_scores[chunk_id]
            metadata["hybrid_score"] = hybrid_score
            metadata["score"] = hybrid_score
            merged[chunk_id] = DocumentChunk(
                id=chunk.id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                metadata=metadata,
            )
            ranked.append((hybrid_score, chunk_id))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [merged[chunk_id] for _score, chunk_id in ranked]

    def create_db(self, stack: DocumentStack):
        if self.skip_indexing:
            return
        chunks = self._split(stack)
        if not chunks:
            chunks = self._split(stack)
        if not chunks:
            raise VectorDBError("Document stack produced no chunks")

        self._index_chunks(chunks)

    def add_stack(self, stack: DocumentStack):
        if self.skip_indexing:
            return
        chunks = self._split(stack)
        if not chunks:
            return
        self._index_chunks(chunks)

    def add_chunk(self, chunk: DocumentChunk):
        self._index_chunks([chunk])

    def retrieve(self, query: str, **kwargs) -> list[DocumentChunk]:
        top_k, num_candidates = self.top_k, self.num_candidates
        search_method = kwargs.pop("search_method", DEFAULT_SEARCH_METHOD)
        if 'top_k' in kwargs:
            top_k = kwargs.pop('top_k')
        if 'num_candidates' in kwargs:
            num_candidates = kwargs.pop('num_candidates')
        return self.search(query, top_k=top_k, num_candidates=num_candidates, search_method=search_method, **kwargs)

    def _vector_search(self, query: str, *, top_k: int | None = None, num_candidates: int | None = None) -> List[DocumentChunk]:
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
        return self._hits_to_chunks(hits)

    def _bm25_search(self, query: str, *, top_k: int | None = None) -> List[DocumentChunk]:
        k = top_k or self.top_k

        body = {
            "query": {
                "match": {
                    self.content_field: {
                        "query": query,
                    }
                }
            },
            "size": k,
        }

        try:
            response = self.client.search(index=self.index_name, body=body)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise VectorDBError("Elasticsearch BM25 search failed") from exc

        hits = response.get("hits", {}).get("hits", [])
        return self._hits_to_chunks(hits)

    def _hybrid_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        num_candidates: int | None = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> List[DocumentChunk]:
        vector_hits = self._vector_search(query, top_k=top_k, num_candidates=num_candidates)
        bm25_hits = self._bm25_search(query, top_k=top_k)
        return self._merge_search_results(vector_hits, bm25_hits, rrf_k=rrf_k)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        num_candidates: int | None = None,
        search_method: str = DEFAULT_SEARCH_METHOD,
        **kwargs,
    ) -> List[DocumentChunk]:
        method = (search_method or DEFAULT_SEARCH_METHOD).lower()
        if method == "vector":
            return self._vector_search(query, top_k=top_k, num_candidates=num_candidates)
        if method == "bm25":
            return self._bm25_search(query, top_k=top_k)
        if method == "hybrid":
            rrf_k = kwargs.pop("rrf_k", DEFAULT_RRF_K)
            return self._hybrid_search(query, top_k=top_k, num_candidates=num_candidates, rrf_k=rrf_k)
        raise VectorDBError(f"Unsupported search method: {search_method}")
