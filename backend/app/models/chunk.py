from pydantic import BaseModel

from app.models.common import DocumentCategory, DocumentSource


class Chunk(BaseModel):
    """One unit of chunked text produced by `ChunkingEngine`.

    `artifact_type` reuses `DocumentCategory` (WORKFLOW/TEST_CASE/ISSUE/
    RELEASE_NOTES/REQUIREMENT) — what kind of content this chunk came
    from, which is what determined the chunking strategy used to
    produce it.
    """

    chunk_id: str
    document_id: str
    feature: str
    artifact_type: DocumentCategory
    document_source: DocumentSource
    source_filename: str
    page_number: int | None = None
    chunk_number: int
    chunk_text: str
