"""Knowledge Assistant: ask + feedback + debug.

`POST /knowledge-assistant/ask` runs the full pipeline — generate a
`generationId`, retrieve context, build the prompt, persist
`user_question.json`/`retrieved_chunks.json`/`prompt.json`, call OpenAI
Chat exactly once, validate the response, persist `response.json`/
`usage.json` — via `KnowledgeAssistantService`, and returns the answer
plus which retrieved documents actually supported it. On failure at any
stage, `error.json` is persisted under that same `generationId` (see
`KnowledgeAssistantService`) and the existing exception-handling
conventions (`app.core.error_handlers`) turn it into the appropriate
HTTP status — never a fabricated answer, never a retry.

`POST /knowledge-assistant/feedback` records a user's judgment on an
already-answered question, keyed by the `generationId` that question's
`ask` call returned — never a question, timestamp, or a second id
invented anywhere. 404s if that `generationId` doesn't exist.

`GET /knowledge-assistant/recent` backs the "Recent Questions and
Answers" list: the newest generations that reached a successful
response, read straight from `GenerationRepository` — never a second
persistence mechanism, and never mock/seeded data. An incomplete or
error-only generation (no `response.json`) is silently skipped, not
fabricated into something it never produced.

`GET /knowledge-assistant/debug/{generation_id}` is read-only: it loads
exactly what was persisted for that generation and reshapes it for
inspection. It never re-runs retrieval, never rebuilds the prompt, and
never calls OpenAI, so what it reports can never drift from what the
model actually saw and said — important since its purpose is answering
"what did the AI actually see (and say, or fail with) for this
question?" It never exposes embeddings, similarity vectors, or the full
prompt text itself (only prompt token/cost information) — a debug
endpoint, not a general read-back of raw prompt content.
"""
from fastapi import APIRouter

from app.core.dependencies import CostCalculatorDep, GenerationRepositoryDep, KnowledgeAssistantServiceDep
from app.models.generation_record import (
    GenerationErrorRecord,
    GenerationRecord,
    RetrievalDiagnosticsRecord,
    RetrievalSourceRecord,
    RetrievedChunkRef,
    UsageRecord,
)
from app.models.hybrid_retrieval import FinalContextChunk, HybridCandidate, VectorQueryDiagnostic
from app.schemas.knowledge_assistant import (
    KnowledgeAssistantAskRequest,
    KnowledgeAssistantAskResponse,
    KnowledgeAssistantCandidateDebugOut,
    KnowledgeAssistantDebugChunkOut,
    KnowledgeAssistantDebugResponse,
    KnowledgeAssistantErrorDebugOut,
    KnowledgeAssistantFeedbackRequest,
    KnowledgeAssistantFeedbackResponse,
    KnowledgeAssistantFinalContextChunkOut,
    KnowledgeAssistantPromptDebugOut,
    KnowledgeAssistantRecentGenerationOut,
    KnowledgeAssistantResponseDebugOut,
    KnowledgeAssistantRetrievalDiagnosticsOut,
    KnowledgeAssistantRetrievalOut,
    KnowledgeAssistantSourceRetrievalOut,
    KnowledgeAssistantUsageDebugOut,
    KnowledgeAssistantUsageOut,
    KnowledgeAssistantVectorQueryDebugOut,
)
from app.services.cost_calculator import CostCalculator

router = APIRouter(tags=["knowledge-assistant"])

_EMPTY_SOURCE_RECORD = RetrievalSourceRecord(
    candidate_retrieved=0, final_chunks=0, merged_chunks=0, documents=[], chunks=[]
)


@router.post("/knowledge-assistant/ask", response_model=KnowledgeAssistantAskResponse)
async def ask_knowledge_assistant(
    payload: KnowledgeAssistantAskRequest,
    knowledge_assistant_service: KnowledgeAssistantServiceDep,
    generation_repository: GenerationRepositoryDep,
) -> KnowledgeAssistantAskResponse:
    """`generationId` in the response is whatever `ask()` generated —
    the caller no longer supplies one. `answer`/`sourceDocuments` come
    straight from `ask()`'s return value (the just-received, just-
    validated OpenAI response) — `usage` is read back from what `ask()`
    persisted, since that's the actual OpenAI-reported cost, not
    something this endpoint computes itself.
    """
    generation_id, answer = knowledge_assistant_service.ask(
        user_question=payload.user_question,
        feature=payload.feature,
        upload_session_id=payload.upload_session_id,
        top_k=payload.top_k,
    )

    usage = generation_repository.get_generation(generation_id).usage

    return KnowledgeAssistantAskResponse(
        generation_id=generation_id,
        question=payload.user_question,
        answer=answer.answer,
        source_documents=answer.source_documents,
        usage=_usage_out(usage),
    )


@router.post("/knowledge-assistant/feedback", response_model=KnowledgeAssistantFeedbackResponse)
async def submit_knowledge_assistant_feedback(
    payload: KnowledgeAssistantFeedbackRequest,
    generation_repository: GenerationRepositoryDep,
) -> KnowledgeAssistantFeedbackResponse:
    """404s (via `NotFoundError`) if `generationId` doesn't correspond
    to a real, already-asked question — feedback is never recorded
    against an id nobody asked a question with. Persistence only: never
    calls OpenAI, never re-runs retrieval.
    """
    generation_repository.get_generation(payload.generation_id)

    record = generation_repository.save_feedback(
        payload.generation_id,
        evaluation=payload.evaluation,
        reason=payload.reason,
        description=payload.description,
    )
    return KnowledgeAssistantFeedbackResponse(generation_id=record.generation_id, evaluation=record.evaluation)


@router.get("/knowledge-assistant/recent", response_model=list[KnowledgeAssistantRecentGenerationOut])
async def list_recent_knowledge_assistant_generations(
    generation_repository: GenerationRepositoryDep,
    limit: int = 4,
) -> list[KnowledgeAssistantRecentGenerationOut]:
    """Newest-first, per `list_generations()`'s own ordering. Walks that
    lightweight listing and loads full detail (`get_generation`) only
    for as many generations as needed to fill `limit`, skipping any
    that never reached a successful response — so one bad or in-flight
    generation never breaks the list, and this never reads more of the
    generation history than it has to.
    """
    recent: list[KnowledgeAssistantRecentGenerationOut] = []
    for item in generation_repository.list_generations():
        if len(recent) >= limit:
            break
        generation = generation_repository.get_generation(item.generation_id)
        if generation.user_question is None or generation.response is None:
            continue
        recent.append(
            KnowledgeAssistantRecentGenerationOut(
                generation_id=generation.generation_id,
                question=generation.user_question.user_question,
                answer=generation.response.response_answer,
                source_documents=generation.response.source_documents,
                created_at=generation.user_question.created_at,
            )
        )
    return recent


@router.get("/knowledge-assistant/debug/{generation_id}", response_model=KnowledgeAssistantDebugResponse)
async def get_knowledge_assistant_debug(
    generation_id: str,
    generation_repository: GenerationRepositoryDep,
    cost_calculator: CostCalculatorDep,
) -> KnowledgeAssistantDebugResponse:
    """Estimated cost is computed live from the persisted pre-call token
    estimate (never persisted itself, exactly like the ask stage this
    debug endpoint predates); actual cost is read straight from what
    `ask()` already persisted to `usage.json` — never recomputed.
    """
    generation = generation_repository.get_generation(generation_id)
    return _debug_response(generation, cost_calculator)


def _debug_response(generation: GenerationRecord, cost_calculator: CostCalculator) -> KnowledgeAssistantDebugResponse:
    retrieved_chunks = generation.retrieved_chunks
    prompt = generation.prompt

    if prompt is not None:
        estimated_cost = cost_calculator.estimate_input_usage(prompt.estimated_tokens)
        prompt_out = KnowledgeAssistantPromptDebugOut(
            prompt_version=prompt.prompt_version,
            input_tokens=prompt.input_tokens,
            estimated_input_tokens=prompt.estimated_tokens,
            estimated_input_cost_inr=estimated_cost.estimated_input_cost.inr,
        )
    else:
        prompt_out = KnowledgeAssistantPromptDebugOut(
            prompt_version="", input_tokens=0, estimated_input_tokens=0, estimated_input_cost_inr=0.0
        )

    response_out = None
    if generation.response is not None and generation.usage is not None:
        response_out = KnowledgeAssistantResponseDebugOut(
            answer=generation.response.response_answer, usage=_usage_debug_out(generation.usage)
        )

    return KnowledgeAssistantDebugResponse(
        generation_id=generation.generation_id,
        question=generation.user_question.user_question if generation.user_question else "",
        retrieval=KnowledgeAssistantRetrievalOut(
            workflow=_source_out(retrieved_chunks.workflow if retrieved_chunks else None),
            historical_test_cases=_source_out(retrieved_chunks.historical_test_cases if retrieved_chunks else None),
            historical_issues=_source_out(retrieved_chunks.historical_issues if retrieved_chunks else None),
            uploaded_documents=_source_out(retrieved_chunks.uploaded_documents if retrieved_chunks else None),
        ),
        retrieval_diagnostics=_retrieval_diagnostics_out(generation.retrieval_diagnostics),
        prompt=prompt_out,
        response=response_out,
        error=_error_out(generation.error),
    )


def _usage_out(usage: UsageRecord) -> KnowledgeAssistantUsageOut:
    return KnowledgeAssistantUsageOut(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        estimated_input_tokens=usage.estimated_input_tokens,
        input_cost_inr=usage.input_cost_inr,
        output_cost_inr=usage.output_cost_inr,
        total_cost_inr=usage.total_cost_inr,
    )


def _usage_debug_out(usage: UsageRecord) -> KnowledgeAssistantUsageDebugOut:
    return KnowledgeAssistantUsageDebugOut(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        estimated_input_tokens=usage.estimated_input_tokens,
        input_cost_inr=usage.input_cost_inr,
        output_cost_inr=usage.output_cost_inr,
        total_cost_inr=usage.total_cost_inr,
        reasoning_tokens=usage.reasoning_tokens,
        cached_tokens=usage.cached_tokens,
        visible_answer_tokens=usage.visible_answer_tokens,
        structured_output_overhead_tokens=usage.structured_output_overhead_tokens,
    )


def _error_out(error: GenerationErrorRecord | None) -> KnowledgeAssistantErrorDebugOut | None:
    if error is None:
        return None
    return KnowledgeAssistantErrorDebugOut(
        error_type=error.error_type,
        error_message=error.error_message,
        stage=error.stage,
        created_at=error.created_at,
    )


def _retrieval_diagnostics_out(
    record: RetrievalDiagnosticsRecord | None,
) -> KnowledgeAssistantRetrievalDiagnosticsOut | None:
    if record is None:
        return None
    diagnostics = record.diagnostics
    return KnowledgeAssistantRetrievalDiagnosticsOut(
        vector_candidate_count=diagnostics.vector_candidate_count,
        bm25_candidate_count=diagnostics.bm25_candidate_count,
        hybrid_candidate_count=diagnostics.hybrid_candidate_count,
        reranked_candidate_count=diagnostics.reranked_candidate_count,
        final_chunk_count=diagnostics.final_chunk_count,
        duplicates_removed=diagnostics.duplicates_removed,
        vector_queries=[_vector_query_out(query) for query in diagnostics.vector_queries],
        candidates=[_candidate_out(candidate) for candidate in diagnostics.candidates],
        final_context=[_final_context_chunk_out(chunk) for chunk in diagnostics.final_context],
        estimated_prompt_tokens=diagnostics.estimated_prompt_tokens,
        vector_retrieval_ms=diagnostics.vector_retrieval_ms,
        bm25_retrieval_ms=diagnostics.bm25_retrieval_ms,
        hybrid_fusion_ms=diagnostics.hybrid_fusion_ms,
        reranking_ms=diagnostics.reranking_ms,
        total_retrieval_ms=diagnostics.total_retrieval_ms,
    )


def _vector_query_out(query: VectorQueryDiagnostic) -> KnowledgeAssistantVectorQueryDebugOut:
    return KnowledgeAssistantVectorQueryDebugOut(
        artifact_type=query.artifact_type,
        requested_n_results=query.requested_n_results,
        available_chunks=query.available_chunks,
        returned_chunks=query.returned_chunks,
    )


def _candidate_out(candidate: HybridCandidate) -> KnowledgeAssistantCandidateDebugOut:
    return KnowledgeAssistantCandidateDebugOut(
        chunk_id=candidate.chunk_id,
        document_id=candidate.document_id,
        document_name=candidate.document_name,
        artifact_type=candidate.artifact_type,
        section_heading=candidate.section_heading,
        vector_rank=candidate.vector_rank,
        vector_score=candidate.vector_score,
        bm25_rank=candidate.bm25_rank,
        bm25_score=candidate.bm25_score,
        rrf_score=candidate.rrf_score,
        reranker_score=candidate.reranker_score,
        final_rank=candidate.final_rank,
        selected=candidate.selected,
    )


def _final_context_chunk_out(chunk: FinalContextChunk) -> KnowledgeAssistantFinalContextChunkOut:
    return KnowledgeAssistantFinalContextChunkOut(
        chunk_id=chunk.chunk_id,
        document_name=chunk.document_name,
        section_heading=chunk.section_heading,
        text=chunk.text,
    )


def _source_out(source: RetrievalSourceRecord | None) -> KnowledgeAssistantSourceRetrievalOut:
    source = source or _EMPTY_SOURCE_RECORD
    return KnowledgeAssistantSourceRetrievalOut(
        candidate_retrieved=source.candidate_retrieved,
        final_chunks=source.final_chunks,
        merged_chunks=source.merged_chunks,
        documents=source.documents,
        chunks=[_chunk_out(chunk) for chunk in source.chunks],
    )


def _chunk_out(chunk: RetrievedChunkRef) -> KnowledgeAssistantDebugChunkOut:
    return KnowledgeAssistantDebugChunkOut(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_name=chunk.document_name,
        artifact_type=chunk.artifact_type,
        section_heading=chunk.section_heading,
        page_number=chunk.page_number,
        similarity_score=chunk.similarity_score,
        chunk_number=chunk.chunk_number,
        merged=chunk.merged,
        merged_chunk_range=chunk.merged_chunk_range,
        merged_chunk_count=chunk.merged_chunk_count,
    )
