from datetime import datetime

from pydantic import BaseModel

from app.models.embedding_preview import EmbeddingPreviewChunk


class UploadHistoryEntry(BaseModel):
    """One completed uploaded-document embedding run, persisted by
    `HistoryService.save_upload_history` as `uploaded_embedding_summary.json`
    inside its own timestamped folder under `backend/data/history/upload/`.

    Same shape as `IndexHistoryEntry`, with `upload_session_id`/
    `uploaded_at` in place of `feature`/`indexed_at` — uploads are
    scoped by session, not by a Source of Truth feature. Only ever
    written after embeddings were generated *and* stored successfully.
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
    index_status: str = "SUCCESS"


class UploadHistoryRecord(BaseModel):
    """One upload embedding run's full persisted detail —
    `uploaded_embedding_summary.json` plus `embedding_preview.json` —
    as returned by `HistoryService.list_upload_history`.
    """

    history_id: str
    summary: UploadHistoryEntry
    embedding_preview: list[EmbeddingPreviewChunk]
