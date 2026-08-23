from datetime import datetime

from app.schemas.common import CamelModel


class IndexHistoryEntryOut(CamelModel):
    feature: str
    documents_indexed: int
    chunks_indexed: int
    embedding_tokens: int
    estimated_embedding_cost: float
    elapsed_seconds: float
    embedding_model: str
    indexed_at: datetime


class IndexingSummaryOut(CamelModel):
    """One row of Document Indexing & Embedding activity for the Usage
    Dashboard — a distinct AI activity type from Test Plan Generation
    (`GenerationSummaryOut`). `activity_type` is included so a future
    response mixing activity types (`TEST_PLAN_GENERATION`,
    `DOCUMENT_INDEXING`, `KNOWLEDGE_ASSISTANT`) can already be told apart
    by shape alone.
    """

    activity_type: str
    history_id: str
    feature: str
    indexed_at: datetime
    embedding_model: str
    documents_indexed: int
    chunks_indexed: int
    embedding_tokens: int
    average_tokens_per_chunk: float
    embedding_cost_usd: float
    embedding_cost_inr: float
    elapsed_seconds: float
    chroma_collection_name: str
    status: str


class IndexingStatisticsOut(CamelModel):
    total_indexing_runs: int
    total_documents_indexed: int
    total_chunks_indexed: int
    total_embedding_tokens: int
    total_embedding_cost_usd: float
    total_embedding_cost_inr: float
