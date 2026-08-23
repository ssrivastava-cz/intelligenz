"""Wraps the OpenAI embeddings API — the Embedding Service layer between
ChunkingEngine and ChromaDB. Isolated here so `IndexService` never talks
to the OpenAI SDK directly, and so tests can substitute a fake client
instead of making real network calls.
"""
from dataclasses import dataclass

import tiktoken
from openai import OpenAI

from app.core.exceptions import ExternalServiceError

# Keeps individual OpenAI requests small/predictable rather than sending
# an entire feature's chunks (potentially thousands) in one request.
_MAX_INPUTS_PER_REQUEST = 100

# OpenAI's embedding models (text-embedding-3-*, ada-002) are all tokenized
# with this encoding; used as a fallback when tiktoken doesn't recognize a
# model name outright.
_FALLBACK_ENCODING = "cl100k_base"


@dataclass
class EmbeddingBatch:
    embeddings: list[list[float]]
    total_tokens: int


class EmbeddingService:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model
        self._encoding = None

    @property
    def model(self) -> str:
        return self._model

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(embeddings=[], total_tokens=0)

        embeddings: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(texts), _MAX_INPUTS_PER_REQUEST):
            batch = texts[start : start + _MAX_INPUTS_PER_REQUEST]
            try:
                response = self._client.embeddings.create(model=self._model, input=batch)
            except Exception as exc:
                raise ExternalServiceError(f"OpenAI embedding request failed: {exc}") from exc
            embeddings.extend(item.embedding for item in response.data)
            total_tokens += response.usage.total_tokens

        return EmbeddingBatch(embeddings=embeddings, total_tokens=total_tokens)

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Counts tokens per text locally via `tiktoken` — no OpenAI API
        call. Lets `IndexService` persist a per-chunk token count without
        an extra embedding request per chunk (the real API only reports
        an aggregate `total_tokens` per request, never a per-input split).
        """
        encoding = self._get_encoding()
        return [len(encoding.encode(text)) for text in texts]

    def _get_encoding(self) -> tiktoken.Encoding:
        if self._encoding is None:
            try:
                self._encoding = tiktoken.encoding_for_model(self._model)
            except KeyError:
                self._encoding = tiktoken.get_encoding(_FALLBACK_ENCODING)
        return self._encoding
