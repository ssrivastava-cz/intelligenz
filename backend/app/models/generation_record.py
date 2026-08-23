"""Persistence models for one Knowledge Assistant generation — a user
question, its retrieved context, the prompt built from it, the AI's
response, optional user feedback, and usage/cost, all correlated by one
caller-supplied `generation_id` (see `app.repositories.generation_repository`).
Deliberately separate from `app.models.generation_history`, which is the
existing, unrelated Test Plan Generator's history — a different feature
with its own persisted shape, untouched by this module.
"""
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from app.models.hybrid_retrieval import RetrievalDiagnostics


class GenerationFeedbackEvaluation(StrEnum):
    """Coarse, explicit user judgment on a generation's response — not
    to be confused with `app.models.common.FeedbackRating` (`up`/`down`),
    which belongs to the separate, existing Test Plan Generator feedback
    feature.
    """

    GOOD = "GOOD"
    BAD = "BAD"


class KnowledgeAssistantErrorStage(StrEnum):
    """Which step of `KnowledgeAssistantService.ask` an error happened
    in — persisted alongside the error so `error.json` can answer "where
    did this fail?" without anyone needing to read logs.
    """

    INPUT_VALIDATION = "INPUT_VALIDATION"
    RETRIEVAL = "RETRIEVAL"
    PROMPT_BUILDING = "PROMPT_BUILDING"
    OPENAI_REQUEST = "OPENAI_REQUEST"
    RESPONSE_VALIDATION = "RESPONSE_VALIDATION"
    PERSISTENCE = "PERSISTENCE"
    UNKNOWN = "UNKNOWN"


class UserQuestionRecord(BaseModel):
    """`user_question.json` — the question as asked, verbatim."""

    generation_id: str
    user_question: str
    created_at: datetime


class RetrievedChunkRef(BaseModel):
    """One retrieved chunk's identity and relevance, for the audit
    record in `retrieved_chunks.json` — never the chunk's embedding
    vector or full text; ChromaDB remains the only store for those.

    `chunk_id`/`document_id`/`document_name`/`similarity_score` are the
    original, always-populated core; everything below is optional,
    additional detail (mirroring `app.schemas.retrieval.RetrievedChunkOut`,
    so this exposes the same shape of information the existing
    `/retrieval/debug` endpoint already does, just persisted) — added
    later for the Knowledge Assistant debug endpoint, so any existing
    caller that only supplies the original four fields is unaffected.
    """

    chunk_id: str
    document_id: str
    document_name: str
    similarity_score: float
    artifact_type: str | None = None
    section_heading: str | None = None
    page_number: int | None = None
    chunk_number: int | None = None
    merged: bool = False
    merged_chunk_range: str | None = None
    merged_chunk_count: int | None = None


class RetrievalSourceRecord(BaseModel):
    """One knowledge source's (Workflow / Historical Test Cases /
    Historical Issues / Uploaded Documents) retrieval detail for one
    generation — pipeline stage counts (how many chunks Retrieve
    Candidate Chunks, Final Chunk Selection, and Merge Adjacent Chunks
    each returned) plus the documents/chunks that source contributed.
    """

    candidate_retrieved: int
    final_chunks: int
    merged_chunks: int
    documents: list[str]
    chunks: list[RetrievedChunkRef]


class RetrievedChunksRecord(BaseModel):
    """`retrieved_chunks.json` — an audit/retrieval record only.
    Embeddings, vectors, and chunk storage/search remain entirely
    ChromaDB's responsibility; nothing here duplicates them.

    `chunks_retrieved`/`documents_retrieved` are the flat totals across
    every knowledge source (the original shape). `workflow`/
    `historical_test_cases`/`historical_issues`/`uploaded_documents`
    are optional, additive per-source breakdowns — `None` for a record
    saved before this detail existed, or by a caller that only supplies
    the flat totals.
    """

    generation_id: str
    chunks_retrieved: list[RetrievedChunkRef]
    documents_retrieved: list[str]
    workflow: RetrievalSourceRecord | None = None
    historical_test_cases: RetrievalSourceRecord | None = None
    historical_issues: RetrievalSourceRecord | None = None
    uploaded_documents: RetrievalSourceRecord | None = None
    created_at: datetime


class PromptRecord(BaseModel):
    """`prompt.json`. `prompt_version` is persisted so different prompt
    versions can be evaluated against each other later.
    """

    generation_id: str
    prompt: str
    input_tokens: int
    estimated_tokens: int
    prompt_version: str
    created_at: datetime


class ResponseRecord(BaseModel):
    """`response.json` — the AI's answer as returned, verbatim, plus
    which retrieved documents actually supported it (the same list
    `KnowledgeAssistantAskResponse.source_documents` returns live —
    persisted here too so a later read, e.g. the "Recent Questions and
    Answers" list, doesn't need the live request/response cycle to
    know what an old answer cited). `source_documents` defaults to `[]`
    for backward compatibility with `response.json` files persisted
    before this field existed.
    """

    generation_id: str
    response_answer: str
    output_tokens: int
    source_documents: list[str] = []
    created_at: datetime


class FeedbackRecord(BaseModel):
    """`feedback.json`. A generation may exist with no feedback yet —
    `description` is optional even for a `BAD` evaluation, and this
    entire file may not exist at all (see `GenerationRepository.get_generation`).
    Persistence only: no automatic evaluation or RAG improvement happens
    here or anywhere in this module.

    `reason` is the frontend's coarse issue category (e.g.
    "MISSING_INFORMATION") for a `BAD` evaluation — optional, and
    `None` for a `GOOD` evaluation, which has no issue to categorize.
    """

    generation_id: str
    evaluation: GenerationFeedbackEvaluation
    reason: str | None = None
    description: str | None = None
    created_at: datetime


class GenerationErrorRecord(BaseModel):
    """`error.json` — persisted whenever any stage of
    `KnowledgeAssistantService.ask` fails, so a failed generation is
    just as traceable by `generation_id` as a successful one. Whatever
    records were already persisted before the failing stage (e.g.
    `user_question.json`, `retrieved_chunks.json`) remain — this is
    additive, never a rollback. `error_message` is never allowed to
    contain secrets/API keys (see
    `KnowledgeAssistantService._persist_error`, which redacts before
    this is ever constructed).
    """

    generation_id: str
    error_type: str
    error_message: str
    stage: KnowledgeAssistantErrorStage
    created_at: datetime


class UsageRecord(BaseModel):
    """`usage.json` — token counts and cost for one generation, in INR
    only (the Knowledge Assistant has no USD-facing UI yet, unlike Test
    Plan Generation's `MoneyAmount`). Intended to eventually feed a
    Knowledge Assistant section of the Usage Dashboard; nothing here
    computes or aggregates that today.

    `output_tokens` is OpenAI's actual billed `completion_tokens` —
    never just the visible answer's own token count. For a reasoning
    model (e.g. GPT-5), it can be dramatically larger, since
    `reasoning_tokens` (below) is billed as output but never appears in
    `response_answer`. `reasoning_tokens`/`cached_tokens` are `None`
    when OpenAI's response didn't include that detail (see
    `CostCalculator.parse_openai_usage_detail`) — never backfilled or
    guessed. `visible_answer_tokens` is a *local* tiktoken count of
    `response_answer` alone, for comparison only — it never replaces
    `output_tokens`. `structured_output_overhead_tokens` is
    `output_tokens` minus `reasoning_tokens` minus
    `visible_answer_tokens` (floored at 0): the remainder attributable
    to Structured Outputs' JSON schema wrapping — an approximation
    (local tiktoken counting of just the answer text isn't guaranteed
    identical to how those same tokens count inside the full JSON
    structure), not an OpenAI-reported figure.
    """

    generation_id: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_input_tokens: int
    input_cost_inr: float
    output_cost_inr: float
    total_cost_inr: float
    generation_time_ms: float
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    visible_answer_tokens: int | None = None
    structured_output_overhead_tokens: int | None = None
    created_at: datetime


class RetrievalDiagnosticsRecord(BaseModel):
    """`retrieval_diagnostics.json` — wraps `RetrievalDiagnostics` (the
    Hybrid Retrieval pipeline's full candidate/scoring/latency detail;
    see `app.models.hybrid_retrieval`) with the same `generation_id` +
    `created_at` envelope every other record here has. Only ever
    written for a generation that went through
    `RetrievalService.retrieve_hybrid` (the Knowledge Assistant) — a
    generation is never re-run to backfill this if it's missing.
    """

    generation_id: str
    diagnostics: RetrievalDiagnostics
    created_at: datetime


class GenerationRecord(BaseModel):
    """The full reconstruction of one generation returned by
    `GenerationRepository.get_generation` — each field is `None` when
    that particular record hasn't been persisted yet (e.g. `feedback`
    before a user rates the answer), never an error.
    """

    generation_id: str
    user_question: UserQuestionRecord | None
    retrieved_chunks: RetrievedChunksRecord | None
    retrieval_diagnostics: RetrievalDiagnosticsRecord | None
    prompt: PromptRecord | None
    response: ResponseRecord | None
    feedback: FeedbackRecord | None
    usage: UsageRecord | None
    error: GenerationErrorRecord | None


class GenerationListItem(BaseModel):
    """One row of `GenerationRepository.list_generations()` — enough to
    render a "Recent Questions and Answers" list without loading every
    generation's full detail. `user_question`/`created_at` are `None`
    only if `user_question.json` itself is missing or unreadable, which
    should not normally happen for a generation created through the
    repository.
    """

    generation_id: str
    user_question: str | None
    created_at: datetime | None
