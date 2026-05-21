"""
LangChain-backed RAG implementations.

This module wires the generic olrag abstractions to LangChain's core
components so a pipeline can be orchestrated through LangChain runnables
while still depending on the library's embedding, chunking, and vector DB
contracts.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

try:  # pragma: no cover - exercised in environments with LangChain installed
    from langchain_core.documents import Document as LCDocument
    from langchain_core.messages import BaseMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable
    from langchain_core.vectorstores import InMemoryVectorStore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "langchain-core is required to use olrag.rag.langchain_rag; "
        "install it via `pip install langchain-core`."
    ) from exc

from olrag.core.document import Chunker, DocumentChunk, DocumentStack
from olrag.core.embedding import EmbeddingModel
from olrag.core.exceptions import InferenceError, RetrievalError, VectorDBError
from olrag.core.inference import InferenceModel
from olrag.core.rag import RAG
from olrag.core.vectordb import VectorDB


def _chunk_to_document(chunk: DocumentChunk) -> LCDocument:
    """Convert an olrag chunk to a LangChain Document with metadata."""
    metadata = {"chunk_id": chunk.id, "doc_id": chunk.doc_id}
    return LCDocument(page_content=chunk.text, metadata=metadata)


class LangChainVectorDB(VectorDB):
    """
    Vector database backed by LangChain's in-memory vector store.

    The database keeps a local cache of chunks so they can be returned
    through the core API after LangChain performs similarity search.
    """

    def __init__(self, model: EmbeddingModel, chunker: Chunker, *, top_k: int = 4):
        super().__init__(model=model)
        self.chunker = chunker
        self.top_k = top_k
        self._vector_store = InMemoryVectorStore()
        self._chunk_cache: Dict[str, DocumentChunk] = {}

    def _add_chunks(self, chunks: Sequence[DocumentChunk]):
        if not chunks:
            raise VectorDBError("No chunks available to add to LangChain vector store")

        documents = [_chunk_to_document(chunk) for chunk in chunks]
        embeddings = [self.model.embed(chunk).embedding for chunk in chunks]
        ids = [chunk.id for chunk in chunks]

        try:
            self._vector_store.add_documents(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
            )
        except Exception as exc:  # pragma: no cover - defensive against LangChain errors
            raise VectorDBError("Failed to add documents to LangChain vector store") from exc

        for chunk in chunks:
            self._chunk_cache[chunk.id] = chunk

    def _split(self, stack: DocumentStack) -> List[DocumentChunk]:
        try:
            return self.chunker.split_doc(stack)
        except Exception as exc:  # pragma: no cover
            raise VectorDBError("Chunking failed while preparing LangChain vector store") from exc

    def create_db(self, stack: DocumentStack):
        chunks = self._split(stack)
        if not chunks:
            raise VectorDBError("Document stack produced no chunks")

        self._vector_store = InMemoryVectorStore()
        self._chunk_cache.clear()
        self._add_chunks(chunks)

    def add_stack(self, stack: DocumentStack):
        chunks = self._split(stack)
        if not chunks:
            return
        self._add_chunks(chunks)

    def add_chunk(self, chunk: DocumentChunk):
        self._add_chunks([chunk])

    def search(self, query: str, *, top_k: int | None = None) -> List[DocumentChunk]:
        if not self._chunk_cache:
            raise VectorDBError("LangChain vector store is empty; call create_db first")

        query_chunk = DocumentChunk(id="__query__", doc_id="__query__", text=query)
        query_embedding = self.model.embed(query_chunk).embedding
        try:
            results = self._vector_store.similarity_search_by_vector(
                query_embedding,
                k=top_k or self.top_k,
            )
        except Exception as exc:  # pragma: no cover
            raise VectorDBError("LangChain vector search failed") from exc

        matched_chunks: List[DocumentChunk] = []
        for doc in results:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id in self._chunk_cache:
                matched_chunks.append(self._chunk_cache[chunk_id])

        return matched_chunks


class LangChainInferenceModel(InferenceModel):
    """
    InferenceModel powered by a LangChain runnable chain.

    The runnable must accept a mapping with the keys ``question`` and
    ``context`` (string containing the concatenated chunk text). The chain
    may return a plain string, a LangChain BaseMessage, or a mapping that
    includes the generated content.
    """

    def __init__(self, runnable: Runnable):
        super().__init__()
        self._runnable = runnable

    @classmethod
    def from_prompt_and_llm(
        cls,
        prompt: ChatPromptTemplate,
        llm: Runnable,
        *,
        parser: Runnable | None = None,
    ) -> "LangChainInferenceModel":
        chain: Runnable = prompt | llm
        if parser is None:
            parser = StrOutputParser()
        chain = chain | parser
        return cls(chain)

    def _format_context(self, context: Sequence[DocumentChunk]) -> str:
        return "\n\n".join(chunk.text for chunk in context)

    def _prepare_payload(self, query: str, context: Sequence[DocumentChunk]) -> dict:
        return {
            "question": query,
            "context": self._format_context(context),
            "chunks": [
                {"id": chunk.id, "doc_id": chunk.doc_id, "text": chunk.text}
                for chunk in context
            ],
        }

    def infer(self, query: str, context: List[DocumentChunk]) -> str:
        payload = self._prepare_payload(query, context)
        try:
            result = self._runnable.invoke(payload)
        except Exception as exc:  # pragma: no cover
            raise InferenceError("LangChain runnable invocation failed") from exc

        if isinstance(result, str):
            return result
        if isinstance(result, BaseMessage):  # pragma: no branch - simple type unwrap
            return result.content
        if isinstance(result, dict):
            for key in ("output_text", "content", "result"):
                value = result.get(key)
                if isinstance(value, str):
                    return value

        raise InferenceError("Unsupported response type returned by LangChain runnable")


class LangChainRAG(RAG):
    """
    Retrieval-Augmented Generation pipeline orchestrated via LangChain.

    The class reuses the generic RAG lifecycle (ingestion through VectorDB)
    but defers retrieval and generation to LangChain-compatible components.
    """

    def __init__(
        self,
        doc_stack: DocumentStack,
        embed_model: EmbeddingModel,
        db: LangChainVectorDB,
        infer_model: LangChainInferenceModel,
        *,
        top_k: int | None = None,
    ):
        if not isinstance(db, LangChainVectorDB):
            raise TypeError("LangChainRAG requires a LangChainVectorDB instance")
        super().__init__(doc_stack=doc_stack, embed_model=embed_model, db=db, infer_model=infer_model)
        self.db: LangChainVectorDB = db
        self._top_k = top_k or db.top_k

    def retrieve(self, query: str) -> List[DocumentChunk]:
        try:
            chunks = self.db.search(query, top_k=self._top_k)
        except VectorDBError as exc:
            raise RetrievalError("Failed to retrieve chunks using LangChain vector DB") from exc

        return chunks
