from pathlib import Path
import pytest
from rag.adapters.document_adapters import *
from rag.rag.pdf_document_reader import *
from rag.core.document import *
from rag import OpenAIEmbeddingModel, LlamaIndexChunker, ElasticsearchVectorDB

stack = DocumentStackFromPDFFolder('./docs', PyPDFExtractor())
print(stack.documents[0].text)

## doc_stack: DocumentStack, embed_model: EmbeddingModel, db: VectorDB, infer_model: InferenceModel


embed_model = OpenAIEmbeddingModel()

chunker = LlamaIndexChunker()

chunks = chunker.split_doc(documents=stack)
vec = embed_model.embed(chunks[0])
db = ElasticsearchVectorDB(model=embed_model, chunker=chunker)

db.add_stack(stack)



