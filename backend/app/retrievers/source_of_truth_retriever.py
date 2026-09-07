"""Retrieves indexed Source of Truth chunks — Workflow Documents,
Historical Test Cases, and Historical Issue Sheets — from the single
persistent ChromaDB collection every feature is indexed into (via
`IndexService`/`VectorStoreService.replace_feature_chunks`). Optionally
bound to one `feature` at construction; artifact-type filtering
(Workflow vs. Test Case vs. Issue) happens per `retrieve()` call via
`filters`, so one instance serves every Source of Truth query
`RetrievalService` needs for a feature — or, when `feature` is `None`,
across every feature in the collection at once.

`feature=None` is how the Knowledge Assistant searches the entire
Source of Truth corpus (see `RetrievalService.retrieve_hybrid`): the
`feature` metadata written at indexing time is untouched and still
returned on every `RetrievedChunk` (see `chunk_mapping.to_retrieved_chunk`)
as provenance — it simply stops being used to restrict *which* chunks
are eligible to match. The Test Plan Generator's `RetrievalService.retrieve()`
always supplies a real `feature`, so its behavior is completely
unaffected by this.
"""
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.base import Retriever
from app.retrievers.chroma_filters import build_where_clause
from app.retrievers.chunk_mapping import to_retrieved_chunk
from app.services.vector_store_service import VectorStoreService


class SourceOfTruthRetriever(Retriever):
    def __init__(self, vector_store: VectorStoreService, feature: str | None = None) -> None:
        self._vector_store = vector_store
        self._feature = feature

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        combined_filters = self._combined_filters(filters)
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
        combined_filters = self._combined_filters(filters)
        return self._vector_store.count_matching_chunks(where=build_where_clause(combined_filters))

    def _combined_filters(self, filters: dict[str, str] | None) -> dict[str, str]:
        """Only restricts by `feature` when one was actually bound at
        construction — `None` means "every feature", never a filter
        clause of `{"feature": None}` (Chroma would reject that outright).
        """
        feature_filter = {"feature": self._feature} if self._feature is not None else {}
        return {**feature_filter, **(filters or {})}
