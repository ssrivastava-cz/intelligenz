from datetime import datetime

from app.models.common import DocumentCategory, DocumentSource
from app.schemas.common import CamelModel


class IndexDebugChunkOut(CamelModel):
    chunk_id: str
    section_heading: str | None
    artifact_type: DocumentCategory
    source_filename: str
    word_count: int
    embedding_tokens: int | None
    embedding_dimension: int
    embedding_exists: bool
    document_source: DocumentSource
    page_number: int | None


class IndexDebugResponse(CamelModel):
    feature: str
    embedding_model: str
    documents_indexed: int
    chunks_indexed: int
    embedding_tokens: int
    estimated_cost: float
    average_tokens_per_chunk: float
    elapsed_time_seconds: float
    indexed_at: datetime
    collection: str
    chunks: list[IndexDebugChunkOut]
