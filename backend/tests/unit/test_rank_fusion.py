"""Unit tests for `ReciprocalRankFusion` — combines the vector search
ranking and the BM25 ranking by rank position, not raw score (see
`app.retrievers.rank_fusion`).
"""
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.bm25_retriever import BM25Match
from app.retrievers.rank_fusion import ReciprocalRankFusion


def _chunk(chunk_id: str, similarity_score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        similarity_score=similarity_score,
        artifact_type="WORKFLOW",
        feature="Contact Log",
        source_filename="a.md",
        section_heading=None,
        page_number=None,
        collection_name="source_of_truth_chunks",
    )


def test_fuse_returns_empty_list_for_no_candidates():
    fusion = ReciprocalRankFusion(k=60)

    assert fusion.fuse([], []) == []


def test_fuse_includes_a_chunk_found_only_by_vector_search():
    fusion = ReciprocalRankFusion(k=60)

    [fused] = fusion.fuse([_chunk("vector-only")], [])

    assert fused.chunk.chunk_id == "vector-only"
    assert fused.diagnostic.vector_rank == 1
    assert fused.diagnostic.bm25_rank is None
    assert fused.diagnostic.bm25_score is None
    assert fused.diagnostic.rrf_score > 0


def test_fuse_includes_a_chunk_found_only_by_bm25():
    fusion = ReciprocalRankFusion(k=60)
    match = BM25Match(chunk=_chunk("bm25-only"), score=3.2)

    [fused] = fusion.fuse([], [match])

    assert fused.chunk.chunk_id == "bm25-only"
    assert fused.diagnostic.bm25_rank == 1
    assert fused.diagnostic.bm25_score == 3.2
    assert fused.diagnostic.vector_rank is None
    assert fused.diagnostic.vector_score is None


def test_fuse_combines_scores_for_a_chunk_found_by_both_methods():
    fusion = ReciprocalRankFusion(k=60)
    vector_chunk = _chunk("shared", similarity_score=0.7)
    match = BM25Match(chunk=_chunk("shared"), score=2.5)

    [fused] = fusion.fuse([vector_chunk], [match])

    assert fused.diagnostic.vector_rank == 1
    assert fused.diagnostic.bm25_rank == 1
    # Combined RRF score is the sum of both contributions.
    expected = (1 / (60 + 1)) + (1 / (60 + 1))
    assert fused.diagnostic.rrf_score == expected
    # The vector-sourced chunk object is kept (it carries a real
    # similarity_score; the BM25-sourced object for the same id never does).
    assert fused.chunk.similarity_score == 0.7


def test_fuse_produces_one_unified_ranking_from_two_different_orderings():
    """Mirrors the example in the task description: C_10 ranks #1 in
    both methods, Contact Workflow and Issue X swap positions between
    them — RRF should still produce one coherent combined ranking with
    C_10 on top.
    """
    fusion = ReciprocalRankFusion(k=60)
    vector_ranking = [_chunk("c10"), _chunk("contact_workflow"), _chunk("issue_x")]
    bm25_ranking = [
        BM25Match(chunk=_chunk("c10"), score=5.0),
        BM25Match(chunk=_chunk("issue_x"), score=3.0),
        BM25Match(chunk=_chunk("contact_workflow"), score=1.0),
    ]

    fused = fusion.fuse(vector_ranking, bm25_ranking)

    assert fused[0].chunk.chunk_id == "c10"


def test_fuse_sorts_by_descending_rrf_score():
    fusion = ReciprocalRankFusion(k=60)
    vector_ranking = [_chunk("first"), _chunk("second"), _chunk("third")]

    fused = fusion.fuse(vector_ranking, [])

    scores = [candidate.diagnostic.rrf_score for candidate in fused]
    assert scores == sorted(scores, reverse=True)


def test_a_lower_k_gives_more_weight_to_top_ranked_positions():
    """RRF's `k` dampens rank sensitivity — a smaller `k` should widen
    the score gap between rank 1 and rank 2."""
    low_k = ReciprocalRankFusion(k=1)
    high_k = ReciprocalRankFusion(k=1000)
    ranking = [_chunk("first"), _chunk("second")]

    low_k_fused = low_k.fuse(ranking, [])
    high_k_fused = high_k.fuse(ranking, [])

    low_k_gap = low_k_fused[0].diagnostic.rrf_score - low_k_fused[1].diagnostic.rrf_score
    high_k_gap = high_k_fused[0].diagnostic.rrf_score - high_k_fused[1].diagnostic.rrf_score
    assert low_k_gap > high_k_gap
