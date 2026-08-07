import json
import time

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from rag import EmbeddingError

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"


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

def chat_completion(client, system_prompt: str, query: str, response_format: BaseModel|None = None) -> str|dict:
    messages = [
        {
            'role': 'system',
            'content': system_prompt,
        },
        {
            'role': 'user',
            'content': query,
        }
    ]
    try:
        if response_format:
            response = call_api(client, messages, call_type='parse', response_format=response_format)
            return json.loads(response.output[0].content[0].text)
        else:
            response = call_api(client, messages, call_type='chat')
            return response.choices[0].message.content
    except Exception as exc:
        raise EmbeddingError()


def call_api(client: OpenAI, input_data: list|str, response_format: BaseModel|None=None, call_type: str ='chat'):
    # client = OpenAI()
    retry_max_attempts = 5
    retry_base_delay_seconds = 0.01
    last_exc: Exception | None = None
    for attempt in range(1, retry_max_attempts + 1):
        try:
            if call_type.lower() == "embed":
                return client.embeddings.create(
                    model=EMBED_MODEL,
                    input=input_data,
                )
            elif call_type.lower() == "parse":
                if not response_format:
                    raise OpenAIError(f'Missing argument response_format.')
                return client.responses.parse(
                    model=CHAT_MODEL,
                    input=input_data,
                    text_format=response_format
                )
            elif call_type.lower() == "chat":
                return client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages = input_data,
                )
        except Exception as exc:  # pragma: no cover - provider/network failure
            last_exc = exc
            if attempt >= retry_max_attempts or not _is_tpm_rate_limit_error(exc):
                raise
            delay = retry_base_delay_seconds * (2 ** (attempt - 1))
            time.sleep(delay)
    if last_exc is not None:
        if _is_tpm_rate_limit_error(last_exc):
            raise EmbeddingError(
                "OpenAI embedding request was rate limited by tokens per minute after retries"
            ) from last_exc
        raise last_exc
    raise EmbeddingError("OpenAI embedding request failed")

