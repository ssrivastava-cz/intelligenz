"""Persistence abstraction for one Knowledge Assistant generation — a
user question, its retrieved context, prompt, response, optional
feedback, and usage/cost, all correlated by one `generation_id`.

`generation_id` is always supplied by the caller (the future
application/business layer, e.g. a `KnowledgeAssistantService`); no
implementation of this interface invents a second id of its own.

Today the only implementation is `FileSystemGenerationRepository`
(`app.repositories.filesystem_generation_repository`), storing each
generation as a directory of JSON files under `backend/database/`. A
future `MongoGenerationRepository` can implement this same interface
against MongoDB instead — the caller depends only on this abstraction,
so it never needs to know which storage backend is actually in use.
"""
from abc import ABC, abstractmethod

from app.models.generation_record import (
    FeedbackRecord,
    GenerationErrorRecord,
    GenerationFeedbackEvaluation,
    GenerationListItem,
    GenerationRecord,
    KnowledgeAssistantErrorStage,
    PromptRecord,
    ResponseRecord,
    RetrievalDiagnosticsRecord,
    RetrievalSourceRecord,
    RetrievedChunkRef,
    RetrievedChunksRecord,
    UsageRecord,
    UserQuestionRecord,
)
from app.models.hybrid_retrieval import RetrievalDiagnostics


class GenerationRepository(ABC):
    @abstractmethod
    def create_generation(self, generation_id: str) -> None:
        """Ensures `generation_id`'s storage location exists. Safe to
        call more than once, and not strictly required before the
        `save_*` methods below — each of those creates it too if it
        isn't there yet.
        """

    @abstractmethod
    def save_user_question(self, generation_id: str, user_question: str) -> UserQuestionRecord:
        """Persists the question as asked, verbatim."""

    @abstractmethod
    def save_retrieved_chunks(
        self,
        generation_id: str,
        chunks_retrieved: list[RetrievedChunkRef],
        documents_retrieved: list[str],
        workflow: RetrievalSourceRecord | None = None,
        historical_test_cases: RetrievalSourceRecord | None = None,
        historical_issues: RetrievalSourceRecord | None = None,
        uploaded_documents: RetrievalSourceRecord | None = None,
    ) -> RetrievedChunksRecord:
        """Persists which chunks/documents fed this generation's
        prompt — an audit record only. Embeddings, vectors, chunk
        storage, and vector search remain ChromaDB's responsibility;
        no implementation of this method may duplicate them. The four
        per-source breakdowns are optional: omit them to persist just
        the flat totals.
        """

    @abstractmethod
    def save_retrieval_diagnostics(
        self, generation_id: str, diagnostics: RetrievalDiagnostics
    ) -> RetrievalDiagnosticsRecord:
        """Persists the Hybrid Retrieval pipeline's full diagnostics for
        one generation — only ever called by the Knowledge Assistant
        path (`retrieve_hybrid`); a generation retrieved via the Test
        Plan Generator's `retrieve()` never has one. Read back verbatim
        by `GET /knowledge-assistant/debug/{generation_id}`, never
        recomputed.
        """

    @abstractmethod
    def save_prompt(
        self,
        generation_id: str,
        prompt: str,
        input_tokens: int,
        estimated_tokens: int,
        prompt_version: str,
    ) -> PromptRecord:
        """Persists the exact prompt sent to the model, and which
        prompt version produced it.
        """

    @abstractmethod
    def save_response(
        self, generation_id: str, response_answer: str, output_tokens: int, source_documents: list[str] | None = None
    ) -> ResponseRecord:
        """Persists the model's answer, verbatim, plus which retrieved
        documents actually supported it (`None`/omitted persists an
        empty list — used by callers, e.g. tests, that don't track this).
        """

    @abstractmethod
    def save_feedback(
        self,
        generation_id: str,
        evaluation: GenerationFeedbackEvaluation,
        reason: str | None = None,
        description: str | None = None,
    ) -> FeedbackRecord:
        """Persists a user's judgment on the response. Persistence
        only — no automatic evaluation or RAG improvement happens as a
        side effect of calling this.
        """

    @abstractmethod
    def save_usage(
        self,
        generation_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        estimated_input_tokens: int,
        input_cost_inr: float,
        output_cost_inr: float,
        total_cost_inr: float,
        generation_time_ms: float,
        reasoning_tokens: int | None = None,
        cached_tokens: int | None = None,
        visible_answer_tokens: int | None = None,
        structured_output_overhead_tokens: int | None = None,
    ) -> UsageRecord:
        """Persists token usage, cost, and timing exactly as given —
        this method computes nothing (e.g. `total_tokens` is not
        derived from `input_tokens + output_tokens` here); the caller
        is responsible for the actual cost/token math. The optional
        fields are OpenAI's detailed usage breakdown plus local
        diagnostics (see `UsageRecord`) — `None` when not available,
        never computed here either.
        """

    @abstractmethod
    def save_error(
        self,
        generation_id: str,
        error_type: str,
        error_message: str,
        stage: KnowledgeAssistantErrorStage,
    ) -> GenerationErrorRecord:
        """Persists that this generation failed, and where. Callers are
        responsible for never passing secrets/API keys/credentials in
        `error_message` — this method persists exactly what it's given,
        the same as every other `save_*` method here.
        """

    @abstractmethod
    def get_generation(self, generation_id: str) -> GenerationRecord:
        """Reconstructs everything persisted for `generation_id`.
        Raises `app.core.exceptions.NotFoundError` if no generation with
        this id exists at all. A record that simply hasn't been saved
        yet (most commonly `feedback`, before a user rates the answer)
        is `None` on the returned `GenerationRecord`, never an error.
        """

    @abstractmethod
    def list_generations(self) -> list[GenerationListItem]:
        """Every known generation's id plus basic metadata, newest
        first by `created_at` where available — enough to power a
        "Recent Questions and Answers" list without loading each
        generation's full detail.
        """

    @abstractmethod
    def list_usage_records(self) -> list[UsageRecord]:
        """Every persisted `usage.json` — only generations that
        actually completed successfully ever have one (`ask()` never
        writes it unless the whole pipeline, including OpenAI, already
        succeeded), so this never fabricates a usage figure for a
        generation that failed. A missing or corrupted `usage.json` is
        excluded, not an error — one bad historical record must never
        break an aggregate view over all of them (e.g. the Usage
        Dashboard). Order is unspecified; callers needing a particular
        order (e.g. newest first) must sort themselves.
        """
