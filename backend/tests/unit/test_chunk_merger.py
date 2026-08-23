"""Unit tests for AdjacentChunkMerger — the Merge Adjacent Chunks stage.
Combines consecutive chunks (by `chunk_number`) from the same source
document (`collection_name` + `document_id`) into one chunk; anything
missing that adjacency metadata is always left unmerged.
"""
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.chunk_merger import AdjacentChunkMerger


def _chunk(
    chunk_id: str,
    similarity_score: float = 0.5,
    document_id: str | None = None,
    chunk_number: int | None = None,
    collection_name: str = "source_of_truth_chunks",
    text: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text if text is not None else f"text-{chunk_id}",
        similarity_score=similarity_score,
        artifact_type="WORKFLOW",
        feature="Appointments",
        source_filename="a.md",
        section_heading=None,
        page_number=None,
        collection_name=collection_name,
        document_id=document_id,
        chunk_number=chunk_number,
    )


def test_merge_returns_empty_list_for_no_chunks():
    merger = AdjacentChunkMerger()

    assert merger.merge([]) == []


def test_merge_combines_two_consecutive_chunks_from_the_same_document():
    chunks = [
        _chunk("a", document_id="doc-1", chunk_number=1, text="First half."),
        _chunk("b", document_id="doc-1", chunk_number=2, text="Second half."),
    ]
    merger = AdjacentChunkMerger()

    merged = merger.merge(chunks)

    assert len(merged) == 1
    assert merged[0].text == "First half.\n\nSecond half."


def test_merge_combines_three_consecutive_chunks_into_one_run():
    chunks = [
        _chunk("a", document_id="doc-1", chunk_number=1, text="One."),
        _chunk("b", document_id="doc-1", chunk_number=2, text="Two."),
        _chunk("c", document_id="doc-1", chunk_number=3, text="Three."),
    ]
    merger = AdjacentChunkMerger()

    merged = merger.merge(chunks)

    assert len(merged) == 1
    assert merged[0].text == "One.\n\nTwo.\n\nThree."


def test_merge_keeps_a_gap_between_non_consecutive_chunk_numbers_separate():
    chunks = [
        _chunk("a", document_id="doc-1", chunk_number=1, text="One."),
        _chunk("c", document_id="doc-1", chunk_number=3, text="Three."),
    ]
    merger = AdjacentChunkMerger()

    merged = merger.merge(chunks)

    assert len(merged) == 2
    assert {chunk.text for chunk in merged} == {"One.", "Three."}


def test_merge_sorts_by_chunk_number_before_detecting_runs_regardless_of_input_order():
    chunks = [
        _chunk("b", document_id="doc-1", chunk_number=2, text="Second half."),
        _chunk("a", document_id="doc-1", chunk_number=1, text="First half."),
    ]
    merger = AdjacentChunkMerger()

    merged = merger.merge(chunks)

    assert len(merged) == 1
    assert merged[0].text == "First half.\n\nSecond half."


def test_merge_keeps_chunks_from_different_documents_separate():
    chunks = [
        _chunk("a", document_id="doc-a", chunk_number=1),
        _chunk("b", document_id="doc-b", chunk_number=1),
    ]
    merger = AdjacentChunkMerger()

    merged = merger.merge(chunks)

    assert len(merged) == 2


def test_merge_keeps_same_document_id_in_different_collections_separate():
    """A chunk id (and a document id) is only unique within the
    collection it was written into — the same document id appearing in
    two different collections (e.g. Source of Truth vs. an upload
    session) is a coincidence, not a real adjacency.
    """
    chunks = [
        _chunk("a", document_id="doc-1", chunk_number=1, collection_name="source_of_truth_chunks"),
        _chunk("b", document_id="doc-1", chunk_number=2, collection_name="uploaded_documents_sess-1"),
    ]
    merger = AdjacentChunkMerger()

    merged = merger.merge(chunks)

    assert len(merged) == 2


def test_merge_never_merges_chunks_missing_document_id():
    chunks = [_chunk("a", chunk_number=1), _chunk("b", chunk_number=2)]
    merger = AdjacentChunkMerger()

    merged = merger.merge(chunks)

    assert len(merged) == 2


def test_merge_never_merges_chunks_missing_chunk_number():
    chunks = [_chunk("a", document_id="doc-1"), _chunk("b", document_id="doc-1")]
    merger = AdjacentChunkMerger()

    merged = merger.merge(chunks)

    assert len(merged) == 2


def test_merge_leaves_a_lone_chunk_from_a_document_unchanged():
    chunks = [_chunk("a", document_id="doc-1", chunk_number=1, similarity_score=0.7, text="Solo.")]
    merger = AdjacentChunkMerger()

    merged = merger.merge(chunks)

    assert merged == chunks


def test_merged_chunk_takes_the_maximum_similarity_score_of_its_run():
    chunks = [
        _chunk("a", document_id="doc-1", chunk_number=1, similarity_score=0.3),
        _chunk("b", document_id="doc-1", chunk_number=2, similarity_score=0.9),
    ]
    merger = AdjacentChunkMerger()

    [merged] = merger.merge(chunks)

    assert merged.similarity_score == 0.9


def test_merge_result_is_sorted_by_similarity_score_descending():
    chunks = [
        _chunk("low", document_id="doc-1", chunk_number=1, similarity_score=0.1),
        _chunk("high", document_id="doc-2", chunk_number=1, similarity_score=0.9),
    ]
    merger = AdjacentChunkMerger()

    merged = merger.merge(chunks)

    assert [chunk.chunk_id for chunk in merged] == ["high", "low"]


def test_merged_chunk_keeps_identifying_fields_from_the_first_chunk_in_the_run():
    chunks = [
        _chunk("a", document_id="doc-1", chunk_number=1),
        _chunk("b", document_id="doc-1", chunk_number=2),
    ]
    merger = AdjacentChunkMerger()

    [merged] = merger.merge(chunks)

    assert merged.chunk_id == "a"
    assert merged.document_id == "doc-1"
    assert merged.chunk_number == 1


def test_merged_chunk_records_every_original_chunk_number_in_the_run():
    chunks = [
        _chunk("a", document_id="doc-1", chunk_number=1),
        _chunk("b", document_id="doc-1", chunk_number=2),
        _chunk("c", document_id="doc-1", chunk_number=3),
    ]
    merger = AdjacentChunkMerger()

    [merged] = merger.merge(chunks)

    assert merged.merged_chunk_numbers == [1, 2, 3]


def test_merged_chunk_numbers_are_sorted_even_when_input_order_is_reversed():
    chunks = [
        _chunk("b", document_id="doc-1", chunk_number=2),
        _chunk("a", document_id="doc-1", chunk_number=1),
    ]
    merger = AdjacentChunkMerger()

    [merged] = merger.merge(chunks)

    assert merged.merged_chunk_numbers == [1, 2]


def test_unmerged_chunk_has_no_merged_chunk_numbers():
    chunks = [_chunk("a", document_id="doc-1", chunk_number=1)]
    merger = AdjacentChunkMerger()

    [result] = merger.merge(chunks)

    assert result.merged_chunk_numbers is None


def test_chunk_left_unmerged_by_a_gap_has_no_merged_chunk_numbers():
    chunks = [
        _chunk("a", document_id="doc-1", chunk_number=1),
        _chunk("c", document_id="doc-1", chunk_number=3),
    ]
    merger = AdjacentChunkMerger()

    merged = merger.merge(chunks)

    assert all(chunk.merged_chunk_numbers is None for chunk in merged)
