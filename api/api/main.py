import os

from fastapi import FastAPI

from api.adapters.rag_adapter import RAGAdapter
from rag import ElasticsearchVectorDB, OpenAIEmbeddingModel, LlamaIndexChunker, RAG

app = FastAPI()
def build_search_db(index_name: str | None = None) -> ElasticsearchVectorDB:
    embed_model = OpenAIEmbeddingModel()
    chunker = LlamaIndexChunker()
    resolved_index_name = index_name or os.environ.get("ES_INDEX_NAME")
    return ElasticsearchVectorDB(model=embed_model, chunker=chunker, index_name=resolved_index_name)

rag_system = build_search_db()
be = RAGAdapter(RAG())
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/search/{query}")
def read_item(query: str | None = None):
    return {"item_id": item_id, "q": q}