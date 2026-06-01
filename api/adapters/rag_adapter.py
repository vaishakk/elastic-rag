from abc import ABC

from rag import RAG, DocumentChunk
from api.core.documents import *

def chunk2doc(chunk: DocumentChunk) -> Document:
    return Document('', chunk.text)

def doc2chunk(doc: Document) -> DocumentChunk:
    return DocumentChunk(id='doc', doc_id='doc', text=chunk.text)

class RAGAdapter():

    def __init__(self, rag_system: RAG):
        self.rag_system = rag_system

    def search(self, query: str, limit: int) -> List[Document]:
        results = self.rag_system.retrieve(query=query, limit=limit)
        result_docs = []
        for result in results:
            result_docs.append(Document('', result.text))
        return result_docs

    def infer(self, query: InferQuery) -> InferenceResults:
        return InferenceResults(
            answer=self.rag_system.infer(query=query.query, context=[doc2chunk(doc) for doc in query.context])
        )