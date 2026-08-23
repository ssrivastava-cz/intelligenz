"""Reciprocal Rank Fusion (RRF) — combines the vector search ranking and
the BM25 ranking into one candidate pool, using *rank position* rather
than raw score. Vector similarity (bounded (0, 1]) and BM25 score
(unbounded, corpus-dependent) live on incomparable scales, so summing
them directly would let whichever score happens to have the larger
numeric range dominate; RRF sidesteps that entirely by only ever looking
at where a chunk placed in each ranking.
"""
from dataclasses import dataclass

from app.models.hybrid_retrieval import HybridCandidate
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.bm25_retriever import BM25Match


@dataclass
class FusedCandidate:
    """One chunk carried through the rest of the hybrid pipeline —
    `chunk` (with its text) for building the eventual prompt/merged
    result, `diagnostic` (metadata + scores, no text) for what gets
    persisted and shown in `GET /knowledge-assistant/debug/{id}`.
    """

    chunk: RetrievedChunk
    diagnostic: HybridCandidate


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        """`k` dampens the influence of rank position — the original RRF
        paper's default of 60 is used app-wide unless overridden (see
        `Settings.rrf_k`), and the paper found results fairly insensitive
        to it across a wide range.
        """
        self._k = k

    def fuse(self, vector_chunks: list[RetrievedChunk], bm25_matches: list[BM25Match]) -> list[FusedCandidate]:
        """`vector_chunks` and `bm25_matches` are each already ranked
        best-first by their own retriever. A chunk appearing in both
        rankings gets one combined `rrf_score` (the sum of its 1/(k+rank)
        contribution from each) and keeps the *vector*-sourced chunk
        object (it carries a real `similarity_score`; the BM25-sourced
        object for the same id never does — see `bm25_retriever.
        _to_retrieved_chunk`). Order is by descending `rrf_score`.
        """
        candidates: dict[str, FusedCandidate] = {}

        for rank, chunk in enumerate(vector_chunks, start=1):
            candidates[chunk.chunk_id] = FusedCandidate(
                chunk=chunk,
                diagnostic=HybridCandidate(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.source_filename,
                    artifact_type=chunk.artifact_type,
                    section_heading=chunk.section_heading,
                    vector_rank=rank,
                    vector_score=chunk.similarity_score,
                    rrf_score=self._rrf_term(rank),
                ),
            )

        for rank, match in enumerate(bm25_matches, start=1):
            existing = candidates.get(match.chunk.chunk_id)
            if existing is not None:
                existing.diagnostic.bm25_rank = rank
                existing.diagnostic.bm25_score = match.score
                existing.diagnostic.rrf_score += self._rrf_term(rank)
            else:
                candidates[match.chunk.chunk_id] = FusedCandidate(
                    chunk=match.chunk,
                    diagnostic=HybridCandidate(
                        chunk_id=match.chunk.chunk_id,
                        document_id=match.chunk.document_id,
                        document_name=match.chunk.source_filename,
                        artifact_type=match.chunk.artifact_type,
                        section_heading=match.chunk.section_heading,
                        bm25_rank=rank,
                        bm25_score=match.score,
                        rrf_score=self._rrf_term(rank),
                    ),
                )

        return sorted(candidates.values(), key=lambda candidate: candidate.diagnostic.rrf_score, reverse=True)

    def _rrf_term(self, rank: int) -> float:
        return 1.0 / (self._k + rank)
