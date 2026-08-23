"""Ties Reranking, the Relevance Threshold, Deduplication, and the
final top-N cutoff together, in that order — the stage between
Reciprocal Rank Fusion and `AdjacentChunkMerger` in the Hybrid Retrieval
pipeline (see `app.services.retrieval_service.RetrievalService.
retrieve_hybrid`). Distinct from the existing `FinalChunkSelector`
(pure top-N by vector similarity, still used unchanged by the Test Plan
Generator's `retrieve()`) rather than replacing it, so that path's
behavior can never be affected by this one.
"""
from dataclasses import dataclass

from app.models.hybrid_retrieval import HybridCandidate
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.chunk_deduplicator import ChunkDeduplicator
from app.retrievers.rank_fusion import FusedCandidate
from app.retrievers.reranker import Reranker


@dataclass
class SelectionResult:
    selected_chunks: list[RetrievedChunk]
    all_candidates: list[HybridCandidate]
    reranked_candidate_count: int
    duplicates_removed: int


class RerankingSelector:
    def __init__(
        self,
        reranker: Reranker,
        deduplicator: ChunkDeduplicator | None = None,
        min_score: float = 0.0,
    ) -> None:
        self._reranker = reranker
        self._deduplicator = deduplicator or ChunkDeduplicator()
        self._min_score = min_score

    def select(self, query_text: str, candidates: list[FusedCandidate], final_chunks: int) -> SelectionResult:
        """Never fills `final_chunks` merely to hit a quota: a candidate
        below `min_score` is excluded regardless of how few (or zero)
        candidates remain afterward — see `Settings.reranker_min_score`.
        `all_candidates` (every fused candidate, not just selected ones)
        is what `RetrievalDiagnostics.candidates` persists, so the debug
        endpoint can show *why* a candidate wasn't picked, not just that
        it wasn't.
        """
        if not candidates:
            return SelectionResult(
                selected_chunks=[], all_candidates=[], reranked_candidate_count=0, duplicates_removed=0
            )

        self._reranker.score(query_text, candidates)

        ranked = sorted(
            candidates,
            key=lambda candidate: (candidate.diagnostic.reranker_score, candidate.diagnostic.rrf_score),
            reverse=True,
        )
        above_threshold = [
            candidate for candidate in ranked if candidate.diagnostic.reranker_score >= self._min_score
        ]
        deduplicated, duplicates_removed = self._deduplicator.deduplicate(above_threshold)

        final = deduplicated[:final_chunks] if final_chunks > 0 else []
        for rank, candidate in enumerate(final, start=1):
            candidate.diagnostic.final_rank = rank
            candidate.diagnostic.selected = True

        return SelectionResult(
            selected_chunks=[candidate.chunk for candidate in final],
            all_candidates=[candidate.diagnostic for candidate in ranked],
            reranked_candidate_count=len(above_threshold),
            duplicates_removed=duplicates_removed,
        )
