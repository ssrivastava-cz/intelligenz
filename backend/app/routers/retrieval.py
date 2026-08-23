from fastapi import APIRouter

from app.core.dependencies import RetrievalServiceDep
from app.models.retrieval_result import SourceRetrievalResult
from app.models.retrieved_chunk import RetrievedChunk
from app.models.similarity_statistics import SimilarityStatistics
from app.retrievers.similarity import compute_similarity_statistics
from app.schemas.retrieval import (
    PipelineStageOut,
    QueryEmbeddingOut,
    RetrievalDebugResponse,
    RetrievedChunkOut,
    SourceRetrievalDebugOut,
)

router = APIRouter(tags=["retrieval"])

_CANDIDATE_RETRIEVAL_STAGE = "Candidate Retrieval"
_FINAL_CHUNK_SELECTION_STAGE = "Final Chunk Selection"
_ADJACENT_CHUNK_MERGE_STAGE = "Adjacent Chunk Merge"


@router.get("/retrieval/debug", response_model=RetrievalDebugResponse)
async def get_retrieval_debug(
    retrieval_service: RetrievalServiceDep,
    query: str,
    feature: str,
    upload_session_id: str | None = None,
    top_k: int | None = None,
) -> RetrievalDebugResponse:
    """Read-only inspection of the Retrieval Pipeline for one query: for
    each knowledge source, what Retrieve Candidate Chunks returned, what
    Final Chunk Selection kept, and what Merge Adjacent Chunks combined
    them into — plus similarity-quality diagnostics computed purely from
    scores the vector search already returned (no extra embedding or
    OpenAI calls). No query rewriting exists yet, so `generatedQueryText`
    passes `query` through unchanged. Makes one real OpenAI call to embed
    the query text — the only way to search at all — but never calls
    OpenAI Chat, never runs a Prompt Builder, and never writes to
    ChromaDB or history. Retrieval behavior is unchanged from
    `retrieval_service.retrieve` — this endpoint only formats what it
    already returns.
    """
    result = retrieval_service.retrieve(
        query_text=query, feature=feature, upload_session_id=upload_session_id, top_k=top_k
    )

    return RetrievalDebugResponse(
        user_query=query,
        generated_query_text=result.query_text,
        query_embedding=QueryEmbeddingOut(
            dimension=result.query_embedding_dimension, token_count=result.query_embedding_tokens
        ),
        workflow=_source_debug_out(result.workflow),
        historical_test_cases=_source_debug_out(result.historical_test_cases),
        historical_issues=_source_debug_out(result.historical_issues),
        uploaded_documents=_source_debug_out(result.uploaded_documents),
    )


def _source_debug_out(source: SourceRetrievalResult) -> SourceRetrievalDebugOut:
    candidate_similarity = compute_similarity_statistics(source.candidate_chunks)
    return SourceRetrievalDebugOut(
        candidate_requested=source.candidate_requested,
        candidate_retrieved=len(source.candidate_chunks),
        candidate_chunks=[_chunk_out(chunk) for chunk in source.candidate_chunks],
        final_requested=source.final_requested,
        final_returned=len(source.final_chunks),
        final_chunks=[_chunk_out(chunk) for chunk in source.final_chunks],
        merged_count=len(source.merged_chunks),
        merged_chunks=[_chunk_out(chunk) for chunk in source.merged_chunks],
        average_similarity=candidate_similarity.average,
        minimum_similarity=candidate_similarity.minimum,
        maximum_similarity=candidate_similarity.maximum,
        pipeline_trace=_pipeline_trace(source, candidate_similarity),
    )


def _pipeline_trace(
    source: SourceRetrievalResult, candidate_similarity: SimilarityStatistics
) -> list[PipelineStageOut]:
    return [
        PipelineStageOut(
            stage=_CANDIDATE_RETRIEVAL_STAGE,
            requested=source.candidate_requested,
            returned=len(source.candidate_chunks),
            average_similarity=candidate_similarity.average,
            minimum_similarity=candidate_similarity.minimum,
            maximum_similarity=candidate_similarity.maximum,
        ),
        PipelineStageOut(
            stage=_FINAL_CHUNK_SELECTION_STAGE,
            requested=source.final_requested,
            returned=len(source.final_chunks),
            average_similarity=None,
            minimum_similarity=None,
            maximum_similarity=None,
        ),
        PipelineStageOut(
            stage=_ADJACENT_CHUNK_MERGE_STAGE,
            requested=None,
            returned=len(source.merged_chunks),
            average_similarity=None,
            minimum_similarity=None,
            maximum_similarity=None,
        ),
    ]


def _chunk_out(chunk: RetrievedChunk) -> RetrievedChunkOut:
    merged_chunk_numbers = chunk.merged_chunk_numbers
    is_merged = merged_chunk_numbers is not None
    return RetrievedChunkOut(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        similarity_score=chunk.similarity_score,
        vector_distance=chunk.vector_distance,
        artifact_type=chunk.artifact_type,
        feature=chunk.feature,
        source_filename=chunk.source_filename,
        section_heading=chunk.section_heading,
        page_number=chunk.page_number,
        collection_name=chunk.collection_name,
        document_id=chunk.document_id,
        chunk_number=chunk.chunk_number,
        merged=is_merged,
        merged_chunk_range=(
            f"{min(merged_chunk_numbers)}–{max(merged_chunk_numbers)}" if is_merged else None
        ),
        merged_chunk_count=len(merged_chunk_numbers) if is_merged else None,
    )
