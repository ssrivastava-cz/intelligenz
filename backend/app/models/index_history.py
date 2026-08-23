from datetime import datetime

from pydantic import BaseModel

from app.models.embedding_preview import EmbeddingPreviewChunk


class IndexHistoryEntry(BaseModel):
    """One completed run of `IndexService.index_feature`, persisted by
    `HistoryService.save_index_history` as `index_summary.json` inside
    its own timestamped folder under `backend/data/history/index/`.

    Only ever written after embeddings were generated *and* stored in
    ChromaDB successfully — `index_status` is always `"SUCCESS"` because
    a failed run never reaches `HistoryService` at all.
    """

    feature: str
    indexed_at: datetime
    embedding_model: str
    documents_indexed: int
    chunks_indexed: int
    embedding_tokens: int
    average_tokens_per_chunk: float
    estimated_embedding_cost: float
    elapsed_seconds: float
    chroma_collection_name: str
    index_status: str = "SUCCESS"


class IndexHistoryRecord(BaseModel):
    """One indexing run's full persisted detail — `index_summary.json`
    plus `embedding_preview.json` — as returned by
    `HistoryService.load_index_history`.
    """

    history_id: str
    summary: IndexHistoryEntry
    embedding_preview: list[EmbeddingPreviewChunk]
