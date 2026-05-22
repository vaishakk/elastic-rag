from pathlib import Path
import pytest
from rag.adapters.document_adapters import *
from rag.rag.pdf_document_reader import *
from rag.core.document import *
from rag import OpenAIEmbeddingModel, LlamaIndexChunker, ElasticsearchVectorDB, RAG
from search_docs import search

stack = DocumentStackFromPDFFolder('./docs', PyPDFExtractor())
embed_model = OpenAIEmbeddingModel()
chunker = LlamaIndexChunker()
db = ElasticsearchVectorDB(model=embed_model, chunker=chunker)

search_engine = RAG(doc_stack=stack, embed_model=embed_model, db=db)

print(search_engine.retrieve('whirlpool'))


