"""API-facing schemas for the Knowledge Assistant's ask/feedback/debug
endpoints (`app.routers.knowledge_assistant`). Deliberately separate
from `app.schemas.generation`/`app.schemas.prompt` (the existing Test
Plan Generator's schemas) — a different feature with its own shape.
"""
from datetime import datetime

from app.models.generation_record import GenerationFeedbackEvaluation
from app.schemas.common import CamelModel


class KnowledgeAssistantAskRequest(CamelModel):
    """Request for `POST /knowledge-assistant/ask`. No `generationId`
    here — the backend is the sole authority for it now (generated at
    the start of `KnowledgeAssistantService.ask`), returned in the
    response instead. `feature` is required because `RetrievalService`
    (reused unchanged) has no cross-feature search mode — every
    retrieval is scoped to one feature, exactly like the Test Plan
    Generator's.
    """

    user_question: str
    feature: str
    upload_session_id: str | None = None
    top_k: int | None = None


class KnowledgeAssistantFeedbackRequest(CamelModel):
    """Request for `POST /knowledge-assistant/feedback`. `generationId`
    must be one already returned by a prior `POST /knowledge-assistant/ask`
    — never a question, timestamp, or frontend-generated id (see
    `GenerationRepository.get_generation`, which 404s on an unknown id).
    `reason` is the frontend's issue category for a `BAD` evaluation
    (e.g. "MISSING_INFORMATION"); omit it for `GOOD`.
    """

    generation_id: str
    evaluation: GenerationFeedbackEvaluation
    reason: str | None = None
    description: str | None = None


class KnowledgeAssistantFeedbackResponse(CamelModel):
    generation_id: str
    evaluation: GenerationFeedbackEvaluation


class KnowledgeAssistantUsageOut(CamelModel):
    """Actual OpenAI usage and cost for one generation — always the
    real reported numbers (via `CostCalculator.calculate_actual_usage_from_openai`),
    never the pre-call local estimate (`estimatedInputTokens` is the
    one exception: carried alongside for comparison, but still sourced
    from the persisted pre-call estimate, not recomputed here).
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_input_tokens: int
    input_cost_inr: float
    output_cost_inr: float
    total_cost_inr: float


class KnowledgeAssistantUsageDebugOut(KnowledgeAssistantUsageOut):
    """Extends the standard usage shape with OpenAI's detailed token
    breakdown plus local diagnostics — debug-endpoint (and Usage
    Dashboard) only, so `POST /knowledge-assistant/ask`'s existing,
    tested response shape never changes. `reasoningTokens`/
    `cachedTokens` are `None` when OpenAI's response didn't include
    that detail; `visibleAnswerTokens` is a *local* tiktoken count of
    the answer text alone, never a substitute for `outputTokens` (the
    real, billed figure); `structuredOutputOverheadTokens` is an
    approximation, not an OpenAI-reported figure. See `UsageRecord`.
    """

    reasoning_tokens: int | None
    cached_tokens: int | None
    visible_answer_tokens: int | None
    structured_output_overhead_tokens: int | None


class KnowledgeAssistantAskResponse(CamelModel):
    """The normal, non-debug response — the answer plus which documents
    actually supported it (never a document that wasn't retrieved; see
    `KnowledgeAssistantService`) and its cost. Never the full prompt or
    raw retrieved chunk contents — those are debug-endpoint-only.
    """

    generation_id: str
    question: str
    answer: str
    source_documents: list[str]
    usage: KnowledgeAssistantUsageOut


class KnowledgeAssistantRecentGenerationOut(CamelModel):
    """One row of `GET /knowledge-assistant/recent` — the "Recent
    Questions and Answers" list, backed by real persisted generations
    (`user_question.json`/`response.json`), never mock/seeded data.
    Only generations that reached a successful response are included;
    an incomplete or error-only generation never appears here.
    """

    generation_id: str
    question: str
    answer: str
    source_documents: list[str]
    created_at: datetime


class KnowledgeAssistantDebugChunkOut(CamelModel):
    """One retrieved chunk, for `GET /knowledge-assistant/debug/{generation_id}`
    — every field `/retrieval/debug` already exposes live, just read
    back from what was actually persisted for this generation. Never
    the embedding vector.
    """

    chunk_id: str
    document_id: str
    document_name: str
    artifact_type: str | None
    section_heading: str | None
    page_number: int | None
    similarity_score: float
    chunk_number: int | None
    merged: bool
    merged_chunk_range: str | None
    merged_chunk_count: int | None


class KnowledgeAssistantSourceRetrievalOut(CamelModel):
    """One knowledge source's retrieval detail, as it was for the
    actual prompt this generation used — never re-run live, so this
    can't drift from what the model actually saw.
    """

    candidate_retrieved: int
    final_chunks: int
    merged_chunks: int
    documents: list[str]
    chunks: list[KnowledgeAssistantDebugChunkOut]


class KnowledgeAssistantRetrievalOut(CamelModel):
    workflow: KnowledgeAssistantSourceRetrievalOut
    historical_test_cases: KnowledgeAssistantSourceRetrievalOut
    historical_issues: KnowledgeAssistantSourceRetrievalOut
    uploaded_documents: KnowledgeAssistantSourceRetrievalOut


class KnowledgeAssistantPromptDebugOut(CamelModel):
    """Estimated (pre-call) token/cost information only — never the
    full prompt text itself; `/knowledge-assistant/debug/{generation_id}`
    is a debug endpoint, not a general-purpose way to read back a
    generation's raw prompt.
    """

    prompt_version: str
    input_tokens: int
    estimated_input_tokens: int
    estimated_input_cost_inr: float


class KnowledgeAssistantResponseDebugOut(CamelModel):
    """The actual AI answer plus its actual usage/cost — `None` on the
    parent `KnowledgeAssistantDebugResponse` when this generation never
    reached a successful response (still in progress, or the OpenAI
    call/validation failed). No `sourceDocuments` here: `response.json`
    persists only `response_answer`/`output_tokens` (matching the
    project's established schema for it) — the normal `/ask` response
    already returns `sourceDocuments` directly from the live, just-
    validated answer, and the debug endpoint's own requirements don't
    call for a second copy of it.
    """

    answer: str
    usage: KnowledgeAssistantUsageDebugOut


class KnowledgeAssistantCandidateDebugOut(CamelModel):
    """One Hybrid Retrieval candidate's full scoring trail — why it was
    (or wasn't) selected. `vectorRank`/`vectorScore` and `bm25Rank`/
    `bm25Score` are `None` when that method never surfaced this chunk at
    all (e.g. a chunk found only by BM25 has no vector score); every
    fused candidate always has an `rrfScore` and a `rerankerScore`.
    """

    chunk_id: str
    document_id: str | None
    document_name: str
    artifact_type: str
    section_heading: str | None
    vector_rank: int | None
    vector_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    rrf_score: float
    reranker_score: float | None
    final_rank: int | None
    selected: bool


class KnowledgeAssistantFinalContextChunkOut(CamelModel):
    """One chunk that actually reached the prompt, text included — the
    only place in the debug response with raw chunk text (candidate-
    level diagnostics are metadata/scores only).
    """

    chunk_id: str
    document_name: str
    section_heading: str | None
    text: str


class KnowledgeAssistantVectorQueryDebugOut(CamelModel):
    """One knowledge source's vector query outcome — `available_chunks`
    is how many chunks matched this exact feature/artifactType filter
    at query time (independent of `requested_n_results`), and
    `returned_chunks` is what ChromaDB actually handed back — the three
    together show whether the vector leg of retrieval under- or
    over-delivered relative to what was asked for.
    """

    artifact_type: str
    requested_n_results: int
    available_chunks: int
    returned_chunks: int


class KnowledgeAssistantRetrievalDiagnosticsOut(CamelModel):
    """The Hybrid Retrieval pipeline's full trace for this generation —
    read back verbatim from what was persisted; retrieval is never
    re-run to produce this. `None` on the parent response for a
    generation that never reached (or predates) hybrid retrieval.
    """

    vector_candidate_count: int
    bm25_candidate_count: int
    hybrid_candidate_count: int
    reranked_candidate_count: int
    final_chunk_count: int
    duplicates_removed: int
    vector_queries: list[KnowledgeAssistantVectorQueryDebugOut]
    candidates: list[KnowledgeAssistantCandidateDebugOut]
    final_context: list[KnowledgeAssistantFinalContextChunkOut]
    estimated_prompt_tokens: int
    vector_retrieval_ms: float
    bm25_retrieval_ms: float
    hybrid_fusion_ms: float
    reranking_ms: float
    total_retrieval_ms: float


class KnowledgeAssistantErrorDebugOut(CamelModel):
    """Present only when this generation failed — for development/
    debugging use only; normal, production-facing responses (the ask
    endpoint's error response) never expose this level of internal
    detail, only a safe message via the existing `ApiError` mechanism.
    """

    error_type: str
    error_message: str
    stage: str
    created_at: datetime


class KnowledgeAssistantDebugResponse(CamelModel):
    generation_id: str
    question: str
    retrieval: KnowledgeAssistantRetrievalOut
    retrieval_diagnostics: KnowledgeAssistantRetrievalDiagnosticsOut | None
    prompt: KnowledgeAssistantPromptDebugOut
    response: KnowledgeAssistantResponseDebugOut | None
    error: KnowledgeAssistantErrorDebugOut | None
