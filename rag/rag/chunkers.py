from __future__ import annotations

import hashlib
from typing import List

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SimpleNodeParser

from rag.core.document import Chunker, DocumentChunk, DocumentStack
from rag.core.exceptions import ChunkingError


class LlamaIndexChunker(Chunker):
    """
    Chunker backed by LlamaIndex's SimpleNodeParser with sensible defaults.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def _chunk_id(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def split_doc(self, documents: DocumentStack) -> List[DocumentChunk]:
        llama_docs: List[LlamaDocument] = []
        for doc_id in documents.documents:
            doc = documents.get_doc_by_id(doc_id)
            if not doc.text.strip():
                continue
            metadata = {
                "doc_id": doc.id,
                "path": str(doc.path),
                "title": doc.title,
            }
            llama_docs.append(LlamaDocument(text=doc.text, metadata=metadata))

        if not llama_docs:
            return []

        try:
            nodes = self._parser.get_nodes_from_documents(llama_docs)
        except Exception as exc:  # pragma: no cover - defensive against parser errors
            raise ChunkingError("Failed to chunk documents using LlamaIndex") from exc

        chunks: List[DocumentChunk] = []
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            doc_id = metadata.get("doc_id") or getattr(node, "ref_doc_id", None)

            text = getattr(node, "text", None)
            if not text and hasattr(node, "get_content"):
                try:
                    text = node.get_content()
                except TypeError:
                    text = node.get_content()  # type: ignore[misc]
            text = text or ""

            chunks.append(
                DocumentChunk(
                    id=self._chunk_id(text),
                    doc_id=doc_id,
                    text=text,
                )
            )

        return chunks
