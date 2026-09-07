from fastapi import APIRouter

from app.core.dependencies import CostCalculatorDep, HistoryServiceDep, IndexServiceDep
from app.schemas.index_service import (
    IndexAllResponse,
    IndexHistoryEntryOut,
    IndexingStatisticsOut,
    IndexingSummaryOut,
)

router = APIRouter(tags=["knowledge-base-indexing"])


@router.post("/index-feature/{feature}", response_model=IndexHistoryEntryOut)
async def index_feature(feature: str, index_service: IndexServiceDep) -> IndexHistoryEntryOut:
    """Runs the full Knowledge Base Indexing pipeline for a feature:
    discover -> parse -> chunk -> embed (OpenAI) -> store (ChromaDB).
    Only Source of Truth documents are indexed; no Test Plan is generated.
    """
    entry = index_service.index_feature(feature)
    return IndexHistoryEntryOut.model_validate(entry)


@router.post("/index-source-of-truth", response_model=IndexAllResponse)
async def index_source_of_truth(index_service: IndexServiceDep) -> IndexAllResponse:
    """Indexes every feature folder under `source_of_truth/` in one call,
    running the exact same per-feature pipeline as
    `POST /index-feature/{feature}` (discover -> parse -> chunk ->
    metadata -> embeddings -> ChromaDB -> persistent BM25 -> history),
    one feature at a time. A feature that fails is reported as `FAILED`
    with its error and does not stop the rest. Returns per-feature
    results plus totals.
    """
    return IndexAllResponse.model_validate(index_service.index_all_features())


@router.get("/index-history", response_model=list[IndexHistoryEntryOut])
async def get_index_history(index_service: IndexServiceDep) -> list[IndexHistoryEntryOut]:
    return [IndexHistoryEntryOut.model_validate(entry) for entry in index_service.list_history()]


@router.get("/index-history/summary", response_model=list[IndexingSummaryOut])
async def get_index_history_summary(
    history_service: HistoryServiceDep, cost_calculator: CostCalculatorDep
) -> list[IndexingSummaryOut]:
    """The Usage Dashboard's Document Indexing & Embedding listing —
    every persisted, successful indexing run with its embedding cost in
    both USD and INR (`cost_calculator` only converts the already-
    computed USD cost; it prices nothing itself), newest first. A
    `DOCUMENT_INDEXING` activity is never a generation's cost — see
    `IndexingSummary`.
    """
    summaries = history_service.list_indexing_summaries(cost_calculator)
    summaries.sort(key=lambda summary: summary.indexed_at, reverse=True)
    return [IndexingSummaryOut.model_validate(summary) for summary in summaries]


@router.get("/index-history/stats", response_model=IndexingStatisticsOut)
async def get_index_history_stats(
    history_service: HistoryServiceDep, cost_calculator: CostCalculatorDep
) -> IndexingStatisticsOut:
    """Usage Dashboard aggregates for Document Indexing & Embedding
    activity — see `HistoryService.indexing_statistics`. Entirely
    separate from `/generation/history/stats`; neither total ever
    includes the other's cost.
    """
    return IndexingStatisticsOut.model_validate(history_service.indexing_statistics(cost_calculator))
