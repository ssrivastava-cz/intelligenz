from datetime import datetime

from pydantic import BaseModel, Field

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


class IndexAllFeatureResult(BaseModel):
    """One feature's outcome inside an `IndexService.index_all_features`
    run — a `SUCCESS` projection of that feature's `IndexHistoryEntry`,
    or a `FAILED` marker carrying the error (no history was written for a
    failed feature, exactly as `index_feature` already guarantees).
    """

    feature: str
    status: str
    documents_indexed: int = 0
    chunks_indexed: int = 0
    embedding_tokens: int = 0
    estimated_embedding_cost: float = 0.0
    elapsed_seconds: float = 0.0
    error: str | None = None

    @classmethod
    def from_entry(cls, entry: IndexHistoryEntry) -> "IndexAllFeatureResult":
        return cls(
            feature=entry.feature,
            status="SUCCESS",
            documents_indexed=entry.documents_indexed,
            chunks_indexed=entry.chunks_indexed,
            embedding_tokens=entry.embedding_tokens,
            estimated_embedding_cost=entry.estimated_embedding_cost,
            elapsed_seconds=entry.elapsed_seconds,
        )

    @classmethod
    def failed(cls, feature: str, *, error: str, elapsed_seconds: float) -> "IndexAllFeatureResult":
        return cls(feature=feature, status="FAILED", error=error, elapsed_seconds=elapsed_seconds)


class IndexAllSummary(BaseModel):
    """The aggregate result of indexing every discovered Source of Truth
    feature in one call (`POST /index-source-of-truth`). Totals are sums
    over `features`; a failed feature contributes zeros to the count
    totals and `1` to `failed_features`.
    """

    total_features: int
    successful_features: int
    failed_features: int
    total_documents_indexed: int
    total_chunks_indexed: int
    total_embedding_tokens: int
    total_estimated_embedding_cost: float
    elapsed_seconds: float
    features: list[IndexAllFeatureResult] = Field(default_factory=list)

    @classmethod
    def from_results(
        cls, results: list[IndexAllFeatureResult], *, elapsed_seconds: float
    ) -> "IndexAllSummary":
        successful = [result for result in results if result.status == "SUCCESS"]
        return cls(
            total_features=len(results),
            successful_features=len(successful),
            failed_features=len(results) - len(successful),
            total_documents_indexed=sum(result.documents_indexed for result in results),
            total_chunks_indexed=sum(result.chunks_indexed for result in results),
            total_embedding_tokens=sum(result.embedding_tokens for result in results),
            total_estimated_embedding_cost=sum(result.estimated_embedding_cost for result in results),
            elapsed_seconds=elapsed_seconds,
            features=results,
        )
