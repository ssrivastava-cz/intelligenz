"""Deduplication — the Source of Truth can contain the same information
repeated across multiple documents (e.g. overlapping content between
`Contact_Workflow` and `Sticket_Workflow`), and sending near-identical
context to the prompt twice wastes tokens without adding information.
Runs after reranking + the relevance threshold, before the top-N cutoff
— see `app.retrievers.reranking_selector.RerankingSelector`.
"""
from app.retrievers.rank_fusion import FusedCandidate
from app.retrievers.text_tokenizer import tokenize

_DEFAULT_SIMILARITY_THRESHOLD = 0.8


class ChunkDeduplicator:
    def __init__(self, similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD) -> None:
        self._threshold = similarity_threshold

    def deduplicate(self, candidates: list[FusedCandidate]) -> tuple[list[FusedCandidate], int]:
        """`candidates` must already be sorted best-first — for each
        near-duplicate group (by token-set Jaccard similarity over
        chunk text), only the first (highest-ranked, so highest
        reranker score) representative is kept. Two chunks from
        different documents/sections rarely share enough vocabulary to
        cross this threshold by accident, so legitimate distinct
        chunks — including true adjacent-context neighbours, which
        `AdjacentChunkMerger` handles separately afterward — are left
        alone. Returns the kept candidates plus how many were dropped.
        """
        kept: list[FusedCandidate] = []
        kept_term_sets: list[set[str]] = []
        removed = 0

        for candidate in candidates:
            terms = set(tokenize(candidate.chunk.text))
            if any(_jaccard_similarity(terms, existing) >= self._threshold for existing in kept_term_sets):
                removed += 1
                continue
            kept.append(candidate)
            kept_term_sets.append(terms)

        return kept, removed


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
