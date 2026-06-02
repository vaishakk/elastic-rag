from typing import List

from rag import RAG, DocumentChunk
from ..core.documents import Document, InferQuery, InferenceResults

def chunk2doc(chunk: DocumentChunk) -> Document:
    return Document(title=chunk.doc_id, text=chunk.text)

def doc2chunk(doc: Document) -> DocumentChunk:
    chunk_id = doc.url.name if doc.url else doc.title
    doc_id = doc.url.stem if doc.url else doc.title
    metadata = {"url": str(doc.url)} if doc.url else None
    return DocumentChunk(id=chunk_id, doc_id=doc_id, text=doc.text, metadata=metadata)

class RAGAdapter():

    def __init__(self, rag_system: RAG):
        self.rag_system = rag_system

    def search(self, query: str, limit: int) -> List[Document]:
        results = self.rag_system.retrieve(query=query, limit=limit)
        return [chunk2doc(result) for result in results]

    def infer(self, query: InferQuery) -> InferenceResults:
        refs = [doc.url for doc in query.context if doc.url is not None]
        answer = self.rag_system.infer(
            query=query.query,
            context=[doc2chunk(doc) for doc in query.context],
        )
        return InferenceResults(answer=answer, refs=refs or None)
