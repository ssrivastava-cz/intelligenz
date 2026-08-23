"""Unit tests for `RerankingSelector` — ties Reranking, the Relevance
Threshold, Deduplication, and the final top-N cutoff together, in that
order. See `app.retrievers.reranking_selector`.
"""
from app.models.hybrid_retrieval import HybridCandidate
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.chunk_deduplicator import ChunkDeduplicator
from app.retrievers.rank_fusion import FusedCandidate
from app.retrievers.reranker import Reranker
from app.retrievers.reranking_selector import RerankingSelector

# For tests about threshold/top-N behavior specifically (not
# deduplication) — the default `ChunkDeduplicator`'s Jaccard threshold
# would otherwise collapse this file's short, similarly-worded test
# fixtures together, which has nothing to do with what those tests
# exercise. An unreachable similarity threshold (Jaccard is <= 1.0)
# makes it a no-op.
_NO_OP_DEDUPLICATOR = ChunkDeduplicator(similarity_threshold=1.1)


def _candidate(chunk_id: str, text: str | None = None, rrf_score: float = 0.5) -> FusedCandidate:
    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        # Distinct default text per chunk_id — `RerankingSelector` runs
        # deduplication by default, so identical text across candidates
        # in a test not specifically about deduplication would silently
        # collapse them to one.
        text=text if text is not None else f"distinct content unique to chunk {chunk_id}",
        similarity_score=0.1,
        artifact_type="WORKFLOW",
        feature="Contact Log",
        source_filename=f"{chunk_id}.md",
        section_heading=None,
        page_number=None,
        collection_name="source_of_truth_chunks",
    )
    return FusedCandidate(
        chunk=chunk,
        diagnostic=HybridCandidate(
            chunk_id=chunk_id,
            document_id=None,
            document_name=f"{chunk_id}.md",
            artifact_type="WORKFLOW",
            section_heading=None,
            rrf_score=rrf_score,
        ),
    )


class _StubReranker(Reranker):
    """Assigns each candidate the score from `scores_by_id`, so tests
    can exercise threshold/dedup/top-N behavior without depending on
    `LexicalReranker`'s actual scoring formula."""

    def __init__(self, scores_by_id: dict[str, float]) -> None:
        self._scores_by_id = scores_by_id

    def score(self, query_text: str, candidates: list[FusedCandidate]) -> None:
        for candidate in candidates:
            candidate.diagnostic.reranker_score = self._scores_by_id[candidate.chunk.chunk_id]


def test_select_returns_nothing_for_an_empty_candidate_pool():
    selector = RerankingSelector(reranker=_StubReranker({}))

    result = selector.select("question", [], final_chunks=5)

    assert result.selected_chunks == []
    assert result.all_candidates == []
    assert result.reranked_candidate_count == 0
    assert result.duplicates_removed == 0


def test_select_orders_selected_chunks_by_reranker_score_descending():
    candidates = [_candidate("low"), _candidate("high"), _candidate("mid")]
    selector = RerankingSelector(reranker=_StubReranker({"low": 0.2, "high": 0.9, "mid": 0.5}), min_score=0.0)

    result = selector.select("question", candidates, final_chunks=3)

    assert [chunk.chunk_id for chunk in result.selected_chunks] == ["high", "mid", "low"]


def test_select_excludes_weak_candidates_below_the_relevance_threshold():
    strong = _candidate("strong")
    weak = _candidate("weak")
    selector = RerankingSelector(reranker=_StubReranker({"strong": 0.5, "weak": 0.02}), min_score=0.1)

    result = selector.select("question", [strong, weak], final_chunks=5)

    assert [chunk.chunk_id for chunk in result.selected_chunks] == ["strong"]
    assert result.reranked_candidate_count == 1


def test_select_returns_no_chunks_when_nothing_clears_the_threshold():
    """Never manufactures an answer from weakly related chunks — if
    nothing is relevant enough, nothing is selected, quota or not."""
    candidates = [_candidate("a"), _candidate("b")]
    selector = RerankingSelector(reranker=_StubReranker({"a": 0.01, "b": 0.02}), min_score=0.5)

    result = selector.select("question", candidates, final_chunks=5)

    assert result.selected_chunks == []
    assert result.reranked_candidate_count == 0


def test_select_does_not_fill_the_quota_when_fewer_candidates_are_relevant():
    """final_chunks=5 requested, but only 2 candidates are genuinely
    relevant — exactly 2 must be returned, not 5."""
    candidates = [_candidate("a"), _candidate("b"), _candidate("c"), _candidate("d"), _candidate("e")]
    scores = {"a": 0.9, "b": 0.8, "c": 0.05, "d": 0.03, "e": 0.01}
    selector = RerankingSelector(reranker=_StubReranker(scores), deduplicator=_NO_OP_DEDUPLICATOR, min_score=0.1)

    result = selector.select("question", candidates, final_chunks=5)

    assert len(result.selected_chunks) == 2
    assert {chunk.chunk_id for chunk in result.selected_chunks} == {"a", "b"}


def test_select_respects_the_top_n_cap_even_when_more_candidates_pass_the_threshold():
    candidates = [_candidate(cid) for cid in "abcde"]
    scores = {cid: 0.9 for cid in "abcde"}
    selector = RerankingSelector(reranker=_StubReranker(scores), deduplicator=_NO_OP_DEDUPLICATOR, min_score=0.1)

    result = selector.select("question", candidates, final_chunks=2)

    assert len(result.selected_chunks) == 2


def test_select_deduplicates_before_applying_the_top_n_cap():
    duplicate_text = "Bridged contacts are merged automatically when matching criteria align."
    a = _candidate("a", text=duplicate_text)
    b = _candidate("b", text=duplicate_text)
    c = _candidate("c", text="Completely different content about something else.")
    selector = RerankingSelector(
        reranker=_StubReranker({"a": 0.9, "b": 0.8, "c": 0.7}),
        deduplicator=ChunkDeduplicator(),
        min_score=0.0,
    )

    result = selector.select("question", [a, b, c], final_chunks=5)

    assert [chunk.chunk_id for chunk in result.selected_chunks] == ["a", "c"]
    assert result.duplicates_removed == 1


def test_select_marks_selected_and_final_rank_on_the_diagnostic_objects():
    a = _candidate("a")
    b = _candidate("b")
    selector = RerankingSelector(reranker=_StubReranker({"a": 0.9, "b": 0.8}), min_score=0.0)

    selector.select("question", [a, b], final_chunks=1)

    assert a.diagnostic.selected is True
    assert a.diagnostic.final_rank == 1
    assert b.diagnostic.selected is False
    assert b.diagnostic.final_rank is None


def test_select_all_candidates_includes_every_candidate_not_just_selected_ones():
    a = _candidate("a")
    b = _candidate("b")
    selector = RerankingSelector(reranker=_StubReranker({"a": 0.9, "b": 0.01}), min_score=0.1)

    result = selector.select("question", [a, b], final_chunks=5)

    assert {candidate.chunk_id for candidate in result.all_candidates} == {"a", "b"}
    assert len(result.selected_chunks) == 1
