"""
RAG pipeline powered by LlamaIndex with auto-merging retrieval.

This module wires the generic olrag abstractions to the LlamaIndex-backed
vector database so that retrieval benefits from hierarchical chunk
aggregation while inference remains user-configurable.
"""

from __future__ import annotations

from typing import List

from olrag.core.document import DocumentChunk, DocumentStack
from olrag.core.embedding import EmbeddingModel
from olrag.core.exceptions import RetrievalError, VectorDBError
from olrag.core.inference import InferenceModel
from olrag.core.rag import RAG
from olrag.rag.vectordb import LlamaIndexVectorDB


class LlamaIndexAutoMergingRAG(RAG):
    """
    Retrieval-Augmented Generation pipeline using LlamaIndex AutoMerge retrieval.

    The pipeline delegates chunk indexing to ``LlamaIndexVectorDB`` and
    surfaces merged chunks for downstream inference.
    """

    def __init__(
        self,
        doc_stack: DocumentStack,
        embed_model: EmbeddingModel,
        db: LlamaIndexVectorDB,
        infer_model: InferenceModel,
        *,
        top_k: int | None = None,
    ):
        if not isinstance(db, LlamaIndexVectorDB):
            raise TypeError("LlamaIndexAutoMergingRAG requires a LlamaIndexVectorDB instance")
        super().__init__(doc_stack=doc_stack, embed_model=embed_model, db=db, infer_model=infer_model)
        self.db: LlamaIndexVectorDB = db
        self._top_k = top_k or db.top_k

    def retrieve(self, query: str) -> List[DocumentChunk]:
        try:
            return self.db.search(query, top_k=self._top_k)
        except VectorDBError as exc:
            raise RetrievalError("Failed to retrieve chunks using LlamaIndex AutoMerge retriever") from exc
