from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag.core.document import Document, DocumentStack
from rag.rag.chunkers import LlamaIndexChunker


@dataclass
class _FakeNode:
    text: str
    metadata: dict


class _FakeParser:
    def __init__(self, nodes):
        self._nodes = nodes

    def get_nodes_from_documents(self, documents):
        return self._nodes


def test_llama_index_chunker_uses_hash_of_chunk_text_for_ids():
    chunker = LlamaIndexChunker(chunk_size=16, chunk_overlap=0)
    chunker._parser = _FakeParser(
        [
            _FakeNode(text="same chunk text", metadata={"doc_id": "doc-1"}),
            _FakeNode(text="same chunk text", metadata={"doc_id": "doc-2"}),
            _FakeNode(text="different chunk text", metadata={"doc_id": "doc-3"}),
        ]
    )

    stack = DocumentStack(
        documents=[
            Document(id="doc-1", path=Path("doc-1.txt"), title="Doc 1", text="x"),
            Document(id="doc-2", path=Path("doc-2.txt"), title="Doc 2", text="y"),
            Document(id="doc-3", path=Path("doc-3.txt"), title="Doc 3", text="z"),
        ]
    )

    chunks = chunker.split_doc(stack)

    assert chunks[0].id == chunks[1].id
    assert chunks[0].id != chunks[2].id
    assert len({chunk.id for chunk in chunks}) == 2
    assert chunks[0].doc_id == "doc-1"
    assert chunks[1].doc_id == "doc-2"
    assert chunks[2].doc_id == "doc-3"
