from fastapi import APIRouter

from app.core.dependencies import CostCalculatorDep, EmbeddingServiceDep, PromptBuilderDep, RetrievalServiceDep
from app.models.prompt_input import PromptBuilderInput
from app.models.retrieval_result import SourceRetrievalResult
from app.retrievers.similarity import compute_similarity_statistics
from app.schemas.cost import EstimatedUsageOut
from app.schemas.prompt import (
    PromptDebugChunkOut,
    PromptDebugResponse,
    PromptLengthOut,
    PromptStatisticsOut,
    RetrievedContextSummaryOut,
    SourceRetrievedContextOut,
)
from app.utils.datetime_utils import utcnow

router = APIRouter(tags=["prompt"])


@router.get("/prompt/debug", response_model=PromptDebugResponse)
async def get_prompt_debug(
    retrieval_service: RetrievalServiceDep,
    prompt_builder: PromptBuilderDep,
    embedding_service: EmbeddingServiceDep,
    cost_calculator: CostCalculatorDep,
    feature: str,
    redmine_id: str,
    redmine_description: str,
    description: str | None = None,
    session_id: str | None = None,
    top_k: int | None = None,
) -> PromptDebugResponse:
    """Read-only inspection of the exact prompt the future Test Plan
    Generator would send to the model, built the same way it will:
    `RetrievalService` retrieves context, `PromptBuilder` formats it —
    no logic duplicated from either. Never calls OpenAI Chat, never
    generates a test plan, never saves generation history, never
    modifies what `RetrievalService` returned. Prompt-size statistics
    are estimated locally via `EmbeddingService.count_tokens`
    (tiktoken) — no OpenAI call. Estimated cost reuses that same token
    count via `CostCalculator`, whose pricing always comes from
    configuration, never a hardcoded value.
    """
    query_text = description or redmine_description
    retrieval_result = retrieval_service.retrieve(
        query_text=query_text, feature=feature, upload_session_id=session_id, top_k=top_k
    )

    prompt_input = PromptBuilderInput(
        feature=feature,
        redmine_ticket=redmine_id,
        redmine_description=redmine_description,
        user_description=description,
        workflow_chunks=retrieval_result.workflow_results,
        historical_test_case_chunks=retrieval_result.test_case_results,
        historical_issue_chunks=retrieval_result.issue_results,
        uploaded_document_chunks=retrieval_result.upload_results,
    )
    prompt_result = prompt_builder.build(prompt_input, generated_at=utcnow())
    sections = prompt_result.sections

    (
        system_instructions_tokens,
        redmine_description_tokens,
        workflow_tokens,
        historical_test_case_tokens,
        historical_issue_tokens,
        uploaded_document_tokens,
        estimated_total_prompt_tokens,
    ) = embedding_service.count_tokens(
        [
            sections.system_instructions,
            sections.redmine_description_section,
            sections.workflow_section,
            sections.historical_test_cases_section,
            sections.historical_issues_section,
            sections.uploaded_documents_section,
            prompt_result.prompt_text,
        ]
    )

    included_chunks = [
        *retrieval_result.workflow_results,
        *retrieval_result.test_case_results,
        *retrieval_result.issue_results,
        *retrieval_result.upload_results,
    ]

    estimated_usage = cost_calculator.estimate_input_usage(estimated_total_prompt_tokens)

    return PromptDebugResponse(
        feature=feature,
        redmine_ticket=redmine_id,
        prompt_version=prompt_result.prompt_version,
        prompt_length=PromptLengthOut(
            characters=len(prompt_result.prompt_text),
            words=len(prompt_result.prompt_text.split()),
            estimated_tokens=estimated_total_prompt_tokens,
        ),
        retrieved_context_summary=RetrievedContextSummaryOut(
            workflow=_source_context_out(retrieval_result.workflow),
            historical_test_cases=_source_context_out(retrieval_result.historical_test_cases),
            historical_issues=_source_context_out(retrieval_result.historical_issues),
            uploaded_documents=_source_context_out(retrieval_result.uploaded_documents),
            total_prompt_chunks=(
                len(retrieval_result.workflow.merged_chunks)
                + len(retrieval_result.historical_test_cases.merged_chunks)
                + len(retrieval_result.historical_issues.merged_chunks)
                + len(retrieval_result.uploaded_documents.merged_chunks)
            ),
        ),
        final_prompt=prompt_result.prompt_text,
        included_chunks=[PromptDebugChunkOut.model_validate(chunk) for chunk in included_chunks],
        estimated_usage=EstimatedUsageOut.model_validate(estimated_usage),
        prompt_statistics=PromptStatisticsOut(
            system_instructions_tokens=system_instructions_tokens,
            redmine_description_tokens=redmine_description_tokens,
            workflow_tokens=workflow_tokens,
            historical_test_case_tokens=historical_test_case_tokens,
            historical_issue_tokens=historical_issue_tokens,
            uploaded_document_tokens=uploaded_document_tokens,
            estimated_total_prompt_tokens=estimated_total_prompt_tokens,
        ),
    )


def _source_context_out(source: SourceRetrievalResult) -> SourceRetrievedContextOut:
    return SourceRetrievedContextOut(
        candidate_retrieved=len(source.candidate_chunks),
        final_chunks=len(source.final_chunks),
        merged_chunks=len(source.merged_chunks),
        average_similarity=compute_similarity_statistics(source.candidate_chunks).average,
    )
