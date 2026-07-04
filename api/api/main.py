import os
from functools import lru_cache
from fastapi import FastAPI

from rag.rag.vectordb import ElasticsearchVectorDB
from rag.rag.embedding import OpenAIEmbeddingModel
from rag.rag.chunkers import LlamaIndexChunker
from rag.core.rag import RAG
from rag.rag.markdown_extractors import DocumentStackFromMarkdownFolder, PlainTextExtractor
from rag import ElasticsearchVectorDB, OpenAIEmbeddingModel, LlamaIndexChunker, RAG
from rag.rag.document_extractors.markdown_extractors import DocumentStackFromMarkdownFolder, PlainTextExtractor

app = FastAPI()


def build_rag_system(index_name: str | None = None) -> RAG:
    # stack = DocumentStackFromPDFFolder('./docs', PyPDFExtractor())
    stack = DocumentStackFromMarkdownFolder('./docs/Markdowns', PlainTextExtractor())
    embed_model = OpenAIEmbeddingModel()
    chunker = LlamaIndexChunker()
    resolved_index_name = index_name or os.environ.get("ES_INDEX_NAME")
    return RAG(
        doc_stack=stack,
        embed_model=embed_model,
        db=ElasticsearchVectorDB(
            model=embed_model,
            chunker=chunker,
            index_name=resolved_index_name,
        )
    )


@lru_cache(maxsize=1)
def get_rag_system() -> RAG:
    return build_rag_system()
rag_system = get_rag_system()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/search/{query}")
def answer_query(query: str | None = None):
    ans = rag_system.retrieve(query)
    return {"answer": ans, "q": query}

if __name__ == "__main__":
    # This block only runs if you do: python main.py
    rag_system = get_rag_system()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
