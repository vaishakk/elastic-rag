import os
from functools import lru_cache
from typing import Annotated, Callable

from fastapi import Depends, FastAPI

from rag import ElasticsearchVectorDB, DictDocumentRepository
from rag.core.rag import RAG
from rag.rag.chunkers import LlamaIndexChunker
from rag.rag.document_extractors.markdown_readers import DocumentStackFromMarkdownFolder
from rag.rag.embedding import OpenAIEmbeddingModel


def build_rag_system(index_name: str | None = None) -> RAG:
    stack = DocumentStackFromMarkdownFolder("./docs/Markdowns", DictDocumentRepository())
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
            skip_indexing=True,
        )
    )


@lru_cache(maxsize=1)
def get_rag_system() -> RAG:
    return build_rag_system()


def create_app(rag_provider: Callable[[], RAG] = get_rag_system) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    @app.get("/search/{query}")
    def answer_query(query: str, rag_system: Annotated[RAG, Depends(rag_provider)]):
        ans = rag_system.retrieve(query)
        return {"answer": ans, "q": query}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
