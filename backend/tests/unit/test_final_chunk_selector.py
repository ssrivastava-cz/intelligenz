"""Unit tests for FinalChunkSelector — the Final Chunk Selection stage.
Today it just keeps the top-scoring `final_chunks` candidates; this is
the seam a future Cross Encoder Reranker would replace.
"""
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.final_chunk_selector import FinalChunkSelector


def _chunk(chunk_id: str, similarity_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        similarity_score=similarity_score,
        artifact_type="WORKFLOW",
        feature="Appointments",
        source_filename="a.md",
        section_heading=None,
        page_number=None,
        collection_name="source_of_truth_chunks",
    )


def test_select_returns_empty_list_for_no_candidates():
    selector = FinalChunkSelector()

    assert selector.select([], final_chunks=5) == []


def test_select_keeps_the_top_scoring_candidates():
    candidates = [_chunk("low", 0.2), _chunk("high", 0.9), _chunk("mid", 0.5)]
    selector = FinalChunkSelector()

    selected = selector.select(candidates, final_chunks=2)

    assert [chunk.chunk_id for chunk in selected] == ["high", "mid"]


def test_select_returns_every_candidate_when_final_chunks_exceeds_the_pool():
    candidates = [_chunk("a", 0.5), _chunk("b", 0.9)]
    selector = FinalChunkSelector()

    selected = selector.select(candidates, final_chunks=10)

    assert len(selected) == 2


def test_select_returns_no_chunks_when_final_chunks_is_zero():
    candidates = [_chunk("a", 0.5)]
    selector = FinalChunkSelector()

    assert selector.select(candidates, final_chunks=0) == []


def test_select_re_sorts_even_when_candidates_arrive_out_of_order():
    candidates = [_chunk("a", 0.1), _chunk("b", 0.8), _chunk("c", 0.4)]
    selector = FinalChunkSelector()

    selected = selector.select(candidates, final_chunks=3)

    assert [chunk.chunk_id for chunk in selected] == ["b", "c", "a"]
