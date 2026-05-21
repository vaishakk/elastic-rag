from __future__ import annotations

from typing import List

from openai import OpenAI

from rag.core.document import DocumentChunk
from rag.core.embedding import Embedding, EmbeddingModel
from rag.core.exceptions import EmbeddingError


class OpenAIEmbeddingModel(EmbeddingModel):
    """Embedding model that uses OpenAI's text-embedding API."""

    def __init__(self, model: str = "text-embedding-3-small"):
        super().__init__()
        self.model_name = model
        try:
            self.client = OpenAI()
        except Exception as exc:  # pragma: no cover
            raise EmbeddingError("Failed to initialise OpenAI client") from exc

    def embed(self, chunk: DocumentChunk) -> Embedding:
        if not chunk.text.strip():
            raise EmbeddingError("Cannot embed empty chunk")

        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=chunk.text,
            )
        except Exception as exc:  # pragma: no cover
            raise EmbeddingError("OpenAI embedding request failed") from exc

        try:
            vector = response.data[0].embedding
        except (AttributeError, IndexError) as exc:
            raise EmbeddingError("Malformed response from OpenAI embeddings API") from exc

        return Embedding(chunk_id=chunk.id, embedding=vector)

    def embed_batch(self, chunks: List[DocumentChunk]) -> List[Embedding]:
        texts = [chunk.text for chunk in chunks]
        if any(not text.strip() for text in texts):
            raise EmbeddingError("Cannot embed empty chunk in batch")

        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
        except Exception as exc:  # pragma: no cover
            raise EmbeddingError("OpenAI batch embedding request failed") from exc

        if len(response.data) != len(chunks):
            raise EmbeddingError("Embedding count mismatch in OpenAI response")

        embeddings: List[Embedding] = []
        for chunk, item in zip(chunks, response.data):
            embeddings.append(Embedding(chunk_id=chunk.id, embedding=item.embedding))

        return embeddings
