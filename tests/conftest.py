from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pandas.core.reshape import tile

from rag import DocumentStackFromPDFFolder, TextExtractor, DocumentChunk, InferenceModel, Embedding, VectorDB, \
    EmbeddingModel, Chunker, RAG, Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PDFTextExtractor:
    pass


class MockPDFExtractor(TextExtractor):

    def extract_text(self, url, **kwargs):
        return 'sample text', 'sample title'

class MockStackFromFolder(DocumentStackFromPDFFolder):

    def __init__(self, folder_url):
        self.documents = [
            Document(id='1', path=Path('path'), title='title', text='sample chunk'),
            Document(id='2', path=Path('path'), title='title', text='sample chunk')
        ]

    def list_pdf_dir(self, folder_url):
        return ['url1', 'url2']


class MockChunker(Chunker):

    def split_doc(self, documents):
        chunks = []
        id = 0
        for doc in documents.documents:
            lines = doc.text.split('\n')
            for line in lines:
                chunks.append(DocumentChunk(id=str(id), doc_id=doc.id, text=line))
                id += 1
        return chunks


class MockEmbedModel(EmbeddingModel):

    def embed(self, chunk):
        return Embedding(chunk.id, [1.2, -3.4, 0.4])


class MockDB(VectorDB):

    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.chunker = MockChunker()
        self.chunks = []

    def add_chunk(self, chunk):
        self.chunks.append(chunk)

    def add_stack(self, stack):
        chunks = self.chunker.split_doc(stack)
        for chunk in chunks:
            self.add_chunk(chunk=chunk)

    def create_db(self, stack):
        chunks = self.chunker.split_doc(stack)
        for chunk in chunks:
            self.add_chunk(chunk=chunk)

    def retrieve(self, query: str, **kwargs) -> list[DocumentChunk]:
        return self.chunks


class MockInfer(InferenceModel):

    def __init__(self):
        super().__init__()

    def infer(self, query, context):
        return f"mock response to {query} based on {''.join([chunk.text for chunk in context])}"


class MockRAG(RAG):

    def __init__(self, doc_stack, db, embed_model, infer_model):
        super().__init__(doc_stack, embed_model, db, infer_model)

    def retrieve(self, query, **kwargs):
        return [DocumentChunk(id='1', doc_id='1', text='sample chunk')]

@pytest.fixture
def embed_model():
    return MockEmbedModel()

@pytest.fixture
def rag():
    return MockRAG(
        doc_stack=MockStackFromFolder('path'),
        embed_model=embed_model,
        infer_model=MockInfer(),
        db=MockDB(embed_model=embed_model),
    )

