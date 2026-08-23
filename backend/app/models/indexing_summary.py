from datetime import datetime

from pydantic import BaseModel

from app.models.cost import MoneyAmount
from app.models.index_history import IndexHistoryEntry


class IndexingSummary(BaseModel):
    """The Usage Dashboard's projection of one persisted `IndexHistoryEntry`
    — a `DOCUMENT_INDEXING` activity, entirely distinct from an AI
    generation. Indexing prices a feature's Source of Truth documents
    into the knowledge base; it is never a generation's own cost (see
    `GenerationSummary`, which separately tracks the *uploaded-document*
    embedding cost tied to one specific generation's upload session —
    a different embedding event from this one). Neither ever contributes
    to the other's totals.
    """

    activity_type: str = "DOCUMENT_INDEXING"
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

    @classmethod
    def from_entry(cls, history_id: str, entry: IndexHistoryEntry, embedding_cost: MoneyAmount) -> "IndexingSummary":
        return cls(
            history_id=history_id,
            feature=entry.feature,
            indexed_at=entry.indexed_at,
            embedding_model=entry.embedding_model,
            documents_indexed=entry.documents_indexed,
            chunks_indexed=entry.chunks_indexed,
            embedding_tokens=entry.embedding_tokens,
            average_tokens_per_chunk=entry.average_tokens_per_chunk,
            embedding_cost_usd=embedding_cost.usd,
            embedding_cost_inr=embedding_cost.inr,
            elapsed_seconds=entry.elapsed_seconds,
            chroma_collection_name=entry.chroma_collection_name,
            status=entry.index_status,
        )
