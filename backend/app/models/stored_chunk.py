from pydantic import BaseModel

from app.models.common import DocumentCategory, DocumentSource


class StoredChunk(BaseModel):
    """One chunk as actually persisted in ChromaDB, read back for the
    Knowledge Base Indexing debug endpoint. Never carries the raw
    embedding vector itself — only whether one exists and its dimension.

    `embedding_tokens` is `None` for chunks indexed before per-chunk token
    counts were persisted — old data, not a bug.
    """

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
