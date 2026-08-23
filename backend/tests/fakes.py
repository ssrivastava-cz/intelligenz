"""Shared test doubles.

OpenAI is the one boundary these tests fake outright — it's a real,
paid external network call. ChromaDB is not faked: it's an embedded
local vector store, so tests use a real `chromadb.EphemeralClient`
instead, consistent with this codebase's preference for real objects
over mocks wherever the dependency isn't actually external.
"""
import time
from types import SimpleNamespace

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


class FakeOpenAIClient:
    """Stands in for `openai.OpenAI`. `embeddings.create` returns a
    deterministic embedding (length-based) and a real tiktoken token
    count per input string — enough shape to exercise `EmbeddingService`
    without any network access. Token counts use the same encoding
    `EmbeddingService.count_tokens` uses locally, so a batch's aggregate
    `total_tokens` here stays consistent with the sum of its per-chunk
    counts, just like the real API is expected to.

    `chat.completions.create` and `beta.chat.completions.parse` (the
    Structured Outputs entry point `GenerationService` actually calls)
    both stand in for a Chat Completion call, sharing one implementation
    — returns whatever `chat_response_content` was configured (or raises
    `chat_fail_with`, for exercising `GenerationService`'s error
    handling, or sets `.refusal` when `chat_refusal` is given), with a
    fixed, caller-controlled token usage so tests can assert
    `CostCalculator` was given exactly those numbers rather than a
    locally re-estimated one. `chat_delay_seconds`, if set, blocks for
    that long before responding — only for tests that need a real
    window to observe state while a generation is still in flight.
    """

    def __init__(
        self,
        dimension: int = 3,
        fail_with: Exception | None = None,
        chat_response_content: str | None = None,
        chat_refusal: str | None = None,
        chat_prompt_tokens: int = 100,
        chat_completion_tokens: int = 50,
        chat_fail_with: Exception | None = None,
        chat_delay_seconds: float = 0.0,
        chat_reasoning_tokens: int | None = None,
        chat_cached_tokens: int | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.chat_calls: list[dict] = []
        self.embeddings = SimpleNamespace(create=self._create_embeddings)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=self._create_chat_completion))
        )
        self._dimension = dimension
        self._fail_with = fail_with
        self._chat_response_content = chat_response_content
        self._chat_refusal = chat_refusal
        self._chat_prompt_tokens = chat_prompt_tokens
        self._chat_completion_tokens = chat_completion_tokens
        self._chat_fail_with = chat_fail_with
        self._chat_delay_seconds = chat_delay_seconds
        # `None` by default — matches a real `usage` object for a model
        # that never returns `completion_tokens_details`/
        # `prompt_tokens_details` at all, not just a zero reasoning/
        # cached count. Only set on the fake's returned `usage` when a
        # test explicitly asks for one, e.g. to reproduce a reasoning
        # model's usage shape (see `CostCalculator.parse_openai_usage_detail`).
        self._chat_reasoning_tokens = chat_reasoning_tokens
        self._chat_cached_tokens = chat_cached_tokens

    def _create_embeddings(self, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append(list(input))
        if self._fail_with is not None:
            raise self._fail_with
        data = [SimpleNamespace(embedding=[float(len(text))] * self._dimension) for text in input]
        total_tokens = sum(len(_ENCODING.encode(text)) for text in input)
        return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=total_tokens))

    def _create_chat_completion(self, model: str, messages: list[dict], **kwargs) -> SimpleNamespace:
        self.chat_calls.append({"model": model, "messages": messages, **kwargs})
        if self._chat_delay_seconds:
            # Only for tests that need a real window to observe
            # in-flight state (e.g. GenerationService's progress
            # tracker) — a blocking sleep, matching how the real,
            # synchronous OpenAI SDK call behaves.
            time.sleep(self._chat_delay_seconds)
        if self._chat_fail_with is not None:
            raise self._chat_fail_with

        message = SimpleNamespace(content=self._chat_response_content, refusal=self._chat_refusal)
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(
            prompt_tokens=self._chat_prompt_tokens,
            completion_tokens=self._chat_completion_tokens,
            total_tokens=self._chat_prompt_tokens + self._chat_completion_tokens,
        )
        if self._chat_reasoning_tokens is not None:
            usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=self._chat_reasoning_tokens)
        if self._chat_cached_tokens is not None:
            usage.prompt_tokens_details = SimpleNamespace(cached_tokens=self._chat_cached_tokens)
        return SimpleNamespace(choices=[choice], usage=usage)
