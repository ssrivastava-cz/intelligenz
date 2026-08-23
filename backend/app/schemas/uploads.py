from datetime import datetime

from app.models.common import DocumentCategory
from app.models.embedding_preview import EmbeddingPreviewChunk
from app.schemas.common import CamelModel
from app.schemas.documents import DocumentOut
from app.schemas.source_of_truth import EmbeddingPreviewChunkOut


class UploadFilesResponse(CamelModel):
    """Response for `POST /uploads` — persists the uploaded files and
    nothing else. Reuses `DocumentOut` (the same shape `/upload-documents`
    already returns) rather than inventing a parallel one.
    """

    upload_session_id: str
    documents_uploaded: int
    uploaded_files: list[DocumentOut]


class UploadEmbeddingPreviewChunkOut(CamelModel):
    """One chunk's projected embedding cost for an uploaded document —
    the upload equivalent of `EmbeddingPreviewChunkOut`, with
    `documentName` in place of `sourceFilename` since this endpoint's
    contract names it that way. The underlying `EmbeddingPreviewChunk`
    model (and the persisted `embedding_preview.json` format) still uses
    `source_filename`, matching Source of Truth exactly — only this
    specific API response renames it at the boundary.
    """

    chunk_id: str
    document_name: str
    section_heading: str | None
    page_number: int | None
    artifact_type: DocumentCategory
    word_count: int
    embedding_tokens: int
    estimated_cost: float
    chunk_text: str

    @classmethod
    def from_preview_chunk(cls, chunk: EmbeddingPreviewChunk) -> "UploadEmbeddingPreviewChunkOut":
        return cls(
            chunk_id=chunk.chunk_id,
            document_name=chunk.source_filename,
            section_heading=chunk.section_heading,
            page_number=chunk.page_number,
            artifact_type=chunk.artifact_type,
            word_count=chunk.word_count,
            embedding_tokens=chunk.embedding_tokens,
            estimated_cost=chunk.estimated_cost,
            chunk_text=chunk.chunk_text,
        )


class UploadEmbeddingPreviewResponse(CamelModel):
    upload_session_id: str
    documents_found: int
    chunks_created: int
    embedding_model: str
    total_embedding_tokens: int
    average_tokens_per_chunk: float
    estimated_embedding_cost: float
    # INR alongside the USD estimate above — converted via
    # `CostCalculator.convert_to_money`, the same currency-conversion
    # path used everywhere else, never a second implementation.
    estimated_embedding_cost_inr: float
    chunks: list[UploadEmbeddingPreviewChunkOut]


class UploadEmbeddingResponse(CamelModel):
    """Response for `POST /uploads/{upload_session_id}/embed` — same
    shape as `IndexHistoryEntryOut`, with `uploadSessionId`/`uploadedAt`
    in place of `feature`/`indexedAt`.
    """

    upload_session_id: str
    uploaded_at: datetime
    embedding_model: str
    documents_indexed: int
    chunks_indexed: int
    embedding_tokens: int
    average_tokens_per_chunk: float
    estimated_embedding_cost: float
    elapsed_seconds: float
    chroma_collection_name: str
    index_status: str


class UploadDebugResponse(CamelModel):
    """Read-only inspection of a persisted upload embedding run —
    reused `EmbeddingPreviewChunkOut` (from Source of Truth) for the
    embedding preview, since Prompt 2 requires this to be in exactly
    the same format.
    """

    upload_session_id: str
    uploaded_at: datetime
    embedding_model: str
    documents_indexed: int
    chunks_indexed: int
    embedding_tokens: int
    average_tokens_per_chunk: float
    estimated_embedding_cost: float
    elapsed_seconds: float
    chroma_collection_name: str
    index_status: str
    chunks: list[EmbeddingPreviewChunkOut]
