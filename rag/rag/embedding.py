from __future__ import annotations

import math
import os
import time
from typing import List

from openai import OpenAI

from rag.core.document import DocumentChunk
from rag.core.embedding import Embedding, EmbeddingModel
from rag.core.exceptions import EmbeddingError
from rag.rag.LLMs.open_ai import *

class OpenAIEmbeddingModel(EmbeddingModel):
    """Embedding model that uses OpenAI's text-embedding API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        client: OpenAI | None = None,
        batch_token_limit: int | None = None,
        batch_chunk_limit: int | None = None,
        retry_max_attempts: int | None = None,
        retry_base_delay_seconds: float | None = None,
    ):
        super().__init__()
        self.model_name = model
        self.batch_token_limit = batch_token_limit if batch_token_limit is not None else self._env_int(
            "OPENAI_EMBED_BATCH_TOKEN_LIMIT",
            6000,
        )
        self.batch_chunk_limit = batch_chunk_limit if batch_chunk_limit is not None else self._env_int(
            "OPENAI_EMBED_BATCH_CHUNK_LIMIT",
            64,
        )
        self.retry_max_attempts = (
            retry_max_attempts if retry_max_attempts is not None else self._env_int("OPENAI_EMBED_RETRY_MAX_ATTEMPTS", 5)
        )
        self.retry_base_delay_seconds = (
            retry_base_delay_seconds
            if retry_base_delay_seconds is not None
            else self._env_float("OPENAI_EMBED_RETRY_BASE_DELAY_SECONDS", 1.0)
        )
        if self.batch_token_limit <= 0:
            raise EmbeddingError("Batch token limit must be positive")
        if self.batch_chunk_limit <= 0:
            raise EmbeddingError("Batch chunk limit must be positive")
        if self.retry_max_attempts <= 0:
            raise EmbeddingError("Retry max attempts must be positive")
        if self.retry_base_delay_seconds < 0:
            raise EmbeddingError("Retry base delay must not be negative")
        try:
            self.client = client or OpenAI()
        except Exception as exc:  # pragma: no cover
            raise EmbeddingError("Failed to initialise OpenAI client") from exc

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None or not value.strip():
            return default
        try:
            return int(value)
        except ValueError as exc:  # pragma: no cover - configuration error
            raise EmbeddingError(f"Environment variable {name} must be an integer") from exc

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.environ.get(name)
        if value is None or not value.strip():
            return default
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - configuration error
            raise EmbeddingError(f"Environment variable {name} must be a number") from exc

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Conservative fallback: over-estimate to keep batches below provider limits.
        return max(1, math.ceil(len(text.encode("utf-8")) / 3))

    @staticmethod
    def _is_length_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            phrase in message
            for phrase in (
                "maximum context length",
                "too many tokens",
                "maximum number of tokens",
                "token limit",
                "input is too long",
                "too long",
            )
        )

    @staticmethod
    def _is_tpm_rate_limit_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            phrase in message
            for phrase in (
                "tokens per minute",
                "token per minute",
                "tokens/minute",
                "tokens/min",
                "rate limit",
                "too many requests",
            )
        )

    def _call_embeddings_api(self, *, input_data):
        last_exc: Exception | None = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return self.client.embeddings.create(
                    model=self.model_name,
                    input=input_data,
                )
            except Exception as exc:  # pragma: no cover - provider/network failure
                last_exc = exc
                if attempt >= self.retry_max_attempts or not self._is_tpm_rate_limit_error(exc):
                    raise
                delay = self.retry_base_delay_seconds * (2 ** (attempt - 1))
                time.sleep(delay)
        if last_exc is not None:
            if self._is_tpm_rate_limit_error(last_exc):
                raise EmbeddingError(
                    "OpenAI embedding request was rate limited by tokens per minute after retries"
                ) from last_exc
            raise last_exc
        raise EmbeddingError("OpenAI embedding request failed")

    def _embed_texts(self, chunks: List[DocumentChunk]) -> List[Embedding]:
        texts = [chunk.text for chunk in chunks]
        try:
            # response = self._call_embeddings_api(input_data=texts)
            response = call_api(self.client, input_data=texts, call_type='embed')
        except EmbeddingError:
            raise
        except Exception as exc:  # pragma: no cover
            if len(chunks) > 1 and self._is_length_error(exc):
                midpoint = len(chunks) // 2
                left = self._embed_texts(chunks[:midpoint])
                right = self._embed_texts(chunks[midpoint:])
                return left + right
            if len(chunks) == 1 and self._is_length_error(exc):
                chunk = chunks[0]
                raise EmbeddingError(
                    f"Chunk {chunk.id} exceeds the model input limit. Reduce chunk size or split the document further."
                ) from exc
            raise EmbeddingError("OpenAI batch embedding request failed") from exc

        if len(response.data) != len(chunks):
            raise EmbeddingError("Embedding count mismatch in OpenAI response")

        embeddings: List[Embedding] = []
        for chunk, item in zip(chunks, response.data):
            embeddings.append(Embedding(chunk_id=chunk.id, embedding=item.embedding))

        return embeddings

    def _batch_chunks(self, chunks: List[DocumentChunk]) -> List[List[DocumentChunk]]:
        batches: List[List[DocumentChunk]] = []
        current_batch: List[DocumentChunk] = []
        current_tokens = 0

        for chunk in chunks:
            if not chunk.text.strip():
                raise EmbeddingError("Cannot embed empty chunk in batch")

            chunk_tokens = self._estimate_tokens(chunk.text)
            if chunk_tokens > self.batch_token_limit:
                raise EmbeddingError(
                    f"Chunk {chunk.id} exceeds the configured embedding batch token limit "
                    f"({chunk_tokens} > {self.batch_token_limit}). Reduce chunk size or split the document further."
                )

            next_batch_too_large = (
                current_batch
                and (
                    len(current_batch) >= self.batch_chunk_limit
                    or current_tokens + chunk_tokens > self.batch_token_limit
                )
            )
            if next_batch_too_large:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(chunk)
            current_tokens += chunk_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def embed(self, chunk: DocumentChunk) -> Embedding:
        if not chunk.text.strip():
            raise EmbeddingError("Cannot embed empty chunk")

        try:
            # response = self._call_embeddings_api(input_data=chunk.text)
            response = call_api(client=self.client, input_data=chunk.text, call_type='embed')
        except EmbeddingError:
            raise
        except Exception as exc:  # pragma: no cover
            raise EmbeddingError("OpenAI embedding request failed") from exc

        try:
            vector = response.data[0].embedding
        except (AttributeError, IndexError) as exc:
            raise EmbeddingError("Malformed response from OpenAI embeddings API") from exc

        return Embedding(chunk_id=chunk.id, embedding=vector)

    def embed_batch(self, chunks: List[DocumentChunk]) -> List[Embedding]:
        if not chunks:
            return []

        embeddings: List[Embedding] = []
        for batch in self._batch_chunks(chunks):
            embeddings.extend(self._embed_texts(batch))
        return embeddings

    def embed_batch_(self, chunks: List[DocumentChunk]) -> List[Embedding]:
        return self.embed_batch(chunks)
