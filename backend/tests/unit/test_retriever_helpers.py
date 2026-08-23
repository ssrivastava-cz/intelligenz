"""Unit tests for the small pure-function helpers shared across the
retriever layer: `build_where_clause` (metadata filtering),
`distance_to_similarity` (score mapping), and
`compute_similarity_statistics` (debug-only aggregation over scores
that already exist — never a second similarity calculation).
"""
import pytest

from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.chroma_filters import build_where_clause
from app.retrievers.similarity import compute_similarity_statistics, distance_to_similarity


def test_build_where_clause_returns_none_for_no_filters():
    assert build_where_clause(None) is None
    assert build_where_clause({}) is None


def test_build_where_clause_returns_a_flat_dict_for_a_single_filter():
    assert build_where_clause({"feature": "Appointments"}) == {"feature": "Appointments"}


def test_build_where_clause_combines_multiple_filters_with_and():
    where = build_where_clause({"feature": "Appointments", "artifactType": "WORKFLOW"})

    assert where == {"$and": [{"feature": "Appointments"}, {"artifactType": "WORKFLOW"}]}


def test_distance_to_similarity_is_one_for_zero_distance():
    assert distance_to_similarity(0.0) == 1.0


def test_distance_to_similarity_decreases_as_distance_increases():
    assert distance_to_similarity(1.0) > distance_to_similarity(2.0) > distance_to_similarity(10.0)


def test_distance_to_similarity_stays_within_zero_to_one_bounds():
    for distance in (0.0, 0.5, 1.0, 8.0, 1000.0):
        score = distance_to_similarity(distance)
        assert 0.0 < score <= 1.0


def _chunk(similarity_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        text="text",
        similarity_score=similarity_score,
        artifact_type="WORKFLOW",
        feature="Appointments",
        source_filename="a.md",
        section_heading=None,
        page_number=None,
        collection_name="source_of_truth_chunks",
    )


def test_compute_similarity_statistics_returns_zeros_for_an_empty_list():
    stats = compute_similarity_statistics([])

    assert stats.average == 0.0
    assert stats.minimum == 0.0
    assert stats.maximum == 0.0


def test_compute_similarity_statistics_matches_the_single_score_for_one_chunk():
    stats = compute_similarity_statistics([_chunk(0.75)])

    assert stats.average == 0.75
    assert stats.minimum == 0.75
    assert stats.maximum == 0.75


def test_compute_similarity_statistics_computes_average_minimum_and_maximum():
    stats = compute_similarity_statistics([_chunk(0.2), _chunk(0.8), _chunk(0.5)])

    assert stats.average == pytest.approx(0.5)
    assert stats.minimum == 0.2
    assert stats.maximum == 0.8


def test_compute_similarity_statistics_never_recomputes_similarity_from_distance():
    """Only reads the already-computed `similarity_score` — never touches
    a chunk's distance, so this can't silently duplicate the one real
    similarity calculation in `distance_to_similarity`.
    """
    chunk = _chunk(0.42)

    stats = compute_similarity_statistics([chunk])

    assert stats.average == chunk.similarity_score
