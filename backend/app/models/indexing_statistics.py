from pydantic import BaseModel


class IndexingStatistics(BaseModel):
    """Usage Dashboard aggregates for `DOCUMENT_INDEXING` activity —
    computed directly from persisted indexing history by
    `HistoryService.indexing_statistics()`. Entirely separate from
    `GenerationStatistics`: an indexing run's cost is never folded into
    a generation's total, and vice versa.
    """

    total_indexing_runs: int
    total_documents_indexed: int
    total_chunks_indexed: int
    total_embedding_tokens: int
    total_embedding_cost_usd: float
    total_embedding_cost_inr: float
