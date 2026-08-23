"""Retrieves indexed Source of Truth chunks — Workflow Documents,
Historical Test Cases, and Historical Issue Sheets — from the single
persistent ChromaDB collection every feature is indexed into (via
`IndexService`/`VectorStoreService.replace_feature_chunks`). Bound to
one `feature` at construction; artifact-type filtering (Workflow vs.
Test Case vs. Issue) happens per `retrieve()` call via `filters`, so one
instance serves every Source of Truth query `RetrievalService` needs for
a feature.
"""
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.base import Retriever
from app.retrievers.chroma_filters import build_where_clause
from app.retrievers.chunk_mapping import to_retrieved_chunk
from app.services.vector_store_service import VectorStoreService


class SourceOfTruthRetriever(Retriever):
    def __init__(self, vector_store: VectorStoreService, feature: str) -> None:
        self._vector_store = vector_store
        self._feature = feature

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        combined_filters = {"feature": self._feature, **(filters or {})}
        matches = self._vector_store.query_similar_chunks(
            query_embedding, top_k, where=build_where_clause(combined_filters)
        )
        return [to_retrieved_chunk(match, self._vector_store.collection_name) for match in matches]

    def count_available(self, filters: dict[str, str] | None = None) -> int:
        """How many chunks currently match this same feature/filter
        combination — the same count `retrieve()`'s own call to
        `query_similar_chunks` already computes internally for its
        `n_results` guard, exposed here so a caller (e.g.
        `RetrievalService`, for its "available candidates" debug
        diagnostics) can report it without duplicating the filter-
        building logic.
        """
        combined_filters = {"feature": self._feature, **(filters or {})}
        return self._vector_store.count_matching_chunks(where=build_where_clause(combined_filters))
