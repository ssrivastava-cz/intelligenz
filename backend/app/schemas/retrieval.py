from app.models.common import DocumentCategory
from app.schemas.common import CamelModel


class RetrievedChunkOut(CamelModel):
    chunk_id: str
    text: str
    similarity_score: float
    vector_distance: float
    artifact_type: DocumentCategory
    feature: str
    source_filename: str
    section_heading: str | None
    page_number: int | None
    collection_name: str
    document_id: str | None
    chunk_number: int | None
    # Merge diagnostics — `merged_chunk_range`/`merged_chunk_count` are
    # only meaningful (non-null) when `merged` is true.
    merged: bool
    merged_chunk_range: str | None
    merged_chunk_count: int | None


class QueryEmbeddingOut(CamelModel):
    dimension: int
    token_count: int


class PipelineStageOut(CamelModel):
    """One stage of a knowledge source's retrieval pipeline, kept
    generic (a name plus counts/similarity stats that don't all have to
    apply) so a future stage — Hybrid Search, Cross Encoder Reranker —
    is just another entry appended to `RetrievalDebugResponse`'s stage
    list, never a reason to change this shape or add new top-level
    fields to `SourceRetrievalDebugOut`.
    """

    stage: str
    requested: int | None
    returned: int
    average_similarity: float | None
    minimum_similarity: float | None
    maximum_similarity: float | None


class SourceRetrievalDebugOut(CamelModel):
    """One knowledge source's full pipeline output: how many chunks were
    requested vs. actually returned at each stage, similarity-quality
    stats over the candidate pool, plus the chunks themselves, so
    retrieval quality can be inspected stage by stage.
    """

    candidate_requested: int
    candidate_retrieved: int
    candidate_chunks: list[RetrievedChunkOut]
    final_requested: int
    final_returned: int
    final_chunks: list[RetrievedChunkOut]
    merged_count: int
    merged_chunks: list[RetrievedChunkOut]
    average_similarity: float
    minimum_similarity: float
    maximum_similarity: float
    pipeline_trace: list[PipelineStageOut]


class RetrievalDebugResponse(CamelModel):
    """Read-only, step-by-step view of the Retrieval Pipeline for one
    query — for `GET /retrieval/debug`, used to check retrieval quality
    before any Prompt Builder or OpenAI Chat call exists. One
    `SourceRetrievalDebugOut` per knowledge source, each independently
    showing Retrieve Candidate Chunks -> Final Chunk Selection -> Merge
    Adjacent Chunks.
    """

    user_query: str
    generated_query_text: str
    query_embedding: QueryEmbeddingOut
    workflow: SourceRetrievalDebugOut
    historical_test_cases: SourceRetrievalDebugOut
    historical_issues: SourceRetrievalDebugOut
    uploaded_documents: SourceRetrievalDebugOut
