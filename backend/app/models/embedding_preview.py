from pydantic import BaseModel

from app.models.common import DocumentCategory


class EmbeddingPreviewChunk(BaseModel):
    """One chunk's projected embedding cost, computed locally via
    tiktoken — no embedding vector included. Shared by the
    `/source-of-truth/{feature}/embedding-preview` debug endpoint and
    `IndexService` (which persists the same data via `HistoryService`
    before generating real embeddings), so the token/cost arithmetic for
    a chunk exists in exactly one place.
    """

    chunk_id: str
    section_heading: str | None
    artifact_type: DocumentCategory
    source_filename: str
    page_number: int | None
    word_count: int
    embedding_tokens: int
    estimated_cost: float
    chunk_text: str
