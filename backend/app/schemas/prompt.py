from app.models.common import DocumentCategory
from app.schemas.common import CamelModel
from app.schemas.cost import EstimatedUsageOut


class PromptDebugChunkOut(CamelModel):
    """Why a piece of context appears in the prompt — for every chunk
    the debug endpoint actually included, not the raw `RetrievedChunk`
    shape, since prompt text and metadata (not the full chunk text
    twice) is what a developer needs here.
    """

    chunk_id: str
    artifact_type: DocumentCategory
    source_filename: str
    similarity_score: float
    section_heading: str | None


class PromptLengthOut(CamelModel):
    characters: int
    words: int
    estimated_tokens: int


class SourceRetrievedContextOut(CamelModel):
    """How much one knowledge source contributed before the prompt was
    constructed — candidates retrieved, how many survived Final Chunk
    Selection, how many remained after Merge Adjacent Chunks (the count
    that actually reached the Prompt Builder), and the average
    similarity of its candidate pool (a pure diagnostic — never used to
    re-rank or filter what's included).
    """

    candidate_retrieved: int
    final_chunks: int
    merged_chunks: int
    average_similarity: float


class RetrievedContextSummaryOut(CamelModel):
    workflow: SourceRetrievedContextOut
    historical_test_cases: SourceRetrievedContextOut
    historical_issues: SourceRetrievedContextOut
    uploaded_documents: SourceRetrievedContextOut
    total_prompt_chunks: int


class PromptStatisticsOut(CamelModel):
    system_instructions_tokens: int
    redmine_description_tokens: int
    workflow_tokens: int
    historical_test_case_tokens: int
    historical_issue_tokens: int
    uploaded_document_tokens: int
    estimated_total_prompt_tokens: int


class PromptDebugResponse(CamelModel):
    """Read-only view of exactly what would be sent to the model for one
    ticket/feature — before any OpenAI Chat call, test plan generation,
    or history write exists.
    """

    feature: str
    redmine_ticket: str
    prompt_version: str
    prompt_length: PromptLengthOut
    retrieved_context_summary: RetrievedContextSummaryOut
    final_prompt: str
    included_chunks: list[PromptDebugChunkOut]
    estimated_usage: EstimatedUsageOut
    prompt_statistics: PromptStatisticsOut
