"""Unit tests for chunk_mapping.to_retrieved_chunk — specifically
mapping (or gracefully omitting) `documentId`/`chunkNumber`, since older
ChromaDB collections indexed before those keys existed won't have them,
and mapping the raw vector distance through for debug diagnostics.
"""
from app.retrievers.chunk_mapping import to_retrieved_chunk
from app.services.vector_store_service import VectorMatch


def _match(distance: float = 0.0, **metadata_overrides) -> VectorMatch:
    metadata = {
        "artifactType": "WORKFLOW",
        "feature": "Appointments",
        "sourceFilename": "onboarding.md",
    }
    metadata.update(metadata_overrides)
    return VectorMatch(chunk_id="chunk-1", chunk_text="Body text.", metadata=metadata, distance=distance)


def test_to_retrieved_chunk_maps_document_id_and_chunk_number_when_present():
    match = _match(documentId="doc-1", chunkNumber=3)

    chunk = to_retrieved_chunk(match, collection_name="source_of_truth_chunks")

    assert chunk.document_id == "doc-1"
    assert chunk.chunk_number == 3


def test_to_retrieved_chunk_leaves_document_id_and_chunk_number_none_when_absent():
    """The exact shape of a chunk indexed before this metadata existed —
    must not raise a KeyError, and must not fabricate a value.
    """
    match = _match()

    chunk = to_retrieved_chunk(match, collection_name="source_of_truth_chunks")

    assert chunk.document_id is None
    assert chunk.chunk_number is None


def test_to_retrieved_chunk_maps_the_raw_vector_distance():
    match = _match(distance=1.5)

    chunk = to_retrieved_chunk(match, collection_name="source_of_truth_chunks")

    assert chunk.vector_distance == 1.5


def test_to_retrieved_chunk_derives_similarity_score_from_the_same_distance():
    """`similarity_score` and `vector_distance` must always describe the
    same underlying vector-search result — one isn't a second,
    independently computed value.
    """
    match = _match(distance=1.0)

    chunk = to_retrieved_chunk(match, collection_name="source_of_truth_chunks")

    assert chunk.vector_distance == 1.0
    assert chunk.similarity_score == 0.5  # 1 / (1 + 1)
