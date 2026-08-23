"""Unit tests for `VectorStoreService.query_similar_chunks`'s safety
guard and HNSW-filtered-search fallback — see the reported bug:
`GET /knowledge-assistant` retrieval against a feature's WORKFLOW chunks
failed with ChromaDB's "Cannot return the results in a contiguous 2D
array. Probably ef or M is too small", even though far more chunks
existed than were ever requested (`n_results`).
"""
import uuid

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.core.exceptions import ExternalServiceError
from app.services.vector_store_service import VectorStoreService


def _make_vector_store(collection_name: str | None = None) -> VectorStoreService:
    name = collection_name or f"test_collection_{uuid.uuid4().hex[:8]}"
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    return VectorStoreService(client=client, collection_name=name)


def _seed(
    vector_store: VectorStoreService,
    count: int,
    feature: str = "Contact and Sticket Log",
    artifact_type: str = "WORKFLOW",
):
    ids = [f"chunk-{i}" for i in range(count)]
    embeddings = [[float(i % 37), float((i * 7) % 37)] for i in range(count)]
    documents = [f"Chunk body text number {i}." for i in range(count)]
    metadatas = [
        {
            "chunkId": ids[i],
            "artifactType": artifact_type,
            "feature": feature,
            "documentSource": "source_of_truth",
            "sourceFilename": f"document-{i}.csv",
            "parserName": "CsvParser",
            "parserVersion": "1.0",
        }
        for i in range(count)
    ]
    vector_store.replace_feature_chunks(
        feature=feature, ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )
    return ids


_WHERE_WORKFLOW = {"$and": [{"feature": "Contact and Sticket Log"}, {"artifactType": "WORKFLOW"}]}


# --- n_results guard: zero / fewer / exactly / more than available ---


def test_query_returns_empty_list_when_zero_chunks_are_available(monkeypatch):
    vector_store = _make_vector_store()
    query_call_count = {"n": 0}
    original_query = vector_store._collection.query

    def spy_query(**kwargs):
        query_call_count["n"] += 1
        return original_query(**kwargs)

    monkeypatch.setattr(vector_store._collection, "query", spy_query)

    matches = vector_store.query_similar_chunks([1.0, 2.0], top_k=20, where=_WHERE_WORKFLOW)

    assert matches == []
    # The guard must skip ChromaDB entirely when nothing matches — no
    # query round trip that could never return anything useful anyway.
    assert query_call_count["n"] == 0


def test_query_returns_all_available_when_fewer_than_requested():
    vector_store = _make_vector_store()
    _seed(vector_store, 5)

    matches = vector_store.query_similar_chunks([1.0, 2.0], top_k=20, where=_WHERE_WORKFLOW)

    assert len(matches) == 5


def test_query_returns_exactly_requested_when_counts_match():
    vector_store = _make_vector_store()
    _seed(vector_store, 20)

    matches = vector_store.query_similar_chunks([1.0, 2.0], top_k=20, where=_WHERE_WORKFLOW)

    assert len(matches) == 20


def test_query_returns_only_requested_when_more_are_available():
    vector_store = _make_vector_store()
    _seed(vector_store, 50)

    matches = vector_store.query_similar_chunks([1.0, 2.0], top_k=20, where=_WHERE_WORKFLOW)

    assert len(matches) == 20


def test_query_with_no_top_k_returns_empty_list_without_querying(monkeypatch):
    vector_store = _make_vector_store()
    _seed(vector_store, 5)
    original_query = vector_store._collection.query
    query_call_count = {"n": 0}

    def spy_query(**kwargs):
        query_call_count["n"] += 1
        return original_query(**kwargs)

    monkeypatch.setattr(vector_store._collection, "query", spy_query)

    assert vector_store.query_similar_chunks([1.0, 2.0], top_k=0, where=_WHERE_WORKFLOW) == []
    assert query_call_count["n"] == 0


# --- Diagnostic logging before every query ---


def test_query_logs_feature_artifact_type_available_and_requested_counts(caplog):
    vector_store = _make_vector_store()
    _seed(vector_store, 7)

    with caplog.at_level("INFO"):
        vector_store.query_similar_chunks([1.0, 2.0], top_k=20, where=_WHERE_WORKFLOW)

    [message] = [m for m in caplog.messages if "Vector query:" in m]
    assert "feature='Contact and Sticket Log'" in message
    assert "artifactType='WORKFLOW'" in message
    assert "available_chunks=7" in message
    assert "requested_n_results=20" in message
    assert "embedding_dimension=2" in message


def test_count_matching_chunks_reports_the_real_available_count():
    vector_store = _make_vector_store()
    _seed(vector_store, 12)

    assert vector_store.count_matching_chunks(where=_WHERE_WORKFLOW) == 12
    assert vector_store.count_matching_chunks(where={"artifactType": "TEST_CASE"}) == 0


# --- HNSW filtered-search failure: fall back to brute-force, never fabricate or crash ---


def test_a_hnsw_filtered_search_failure_falls_back_to_brute_force_search(monkeypatch):
    """Reproduces the reported bug directly: ChromaDB's filtered kNN
    search can raise "Cannot return the results in a contiguous 2D
    array. Probably ef or M is too small" even when far more chunks
    exist than requested — `query_similar_chunks` must recover via an
    exact, local similarity search instead of surfacing this as a
    retrieval failure.
    """
    vector_store = _make_vector_store()
    _seed(vector_store, 30)

    def failing_query(**kwargs):
        raise RuntimeError("Cannot return the results in a contigious 2D array. Probably ef or M is too small")

    monkeypatch.setattr(vector_store._collection, "query", failing_query)

    matches = vector_store.query_similar_chunks([1.0, 2.0], top_k=5, where=_WHERE_WORKFLOW)

    assert len(matches) == 5


def test_the_brute_force_fallback_returns_the_true_nearest_neighbors():
    vector_store = _make_vector_store()
    # Three chunks with deliberately distinct, easy-to-reason-about
    # embeddings so the correct nearest-neighbor ordering is unambiguous.
    ids = ["far", "near", "mid"]
    embeddings = [[100.0, 100.0], [1.0, 1.0], [10.0, 10.0]]
    documents = ["Far chunk.", "Near chunk.", "Mid chunk."]
    metadatas = [
        {
            "chunkId": cid,
            "artifactType": "WORKFLOW",
            "feature": "Contact and Sticket Log",
            "documentSource": "source_of_truth",
            "sourceFilename": f"{cid}.csv",
            "parserName": "CsvParser",
            "parserVersion": "1.0",
        }
        for cid in ids
    ]
    vector_store.replace_feature_chunks(
        feature="Contact and Sticket Log", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )

    matches = vector_store._brute_force_query([0.0, 0.0], top_k=3, where=_WHERE_WORKFLOW)

    assert [match.chunk_id for match in matches] == ["near", "mid", "far"]
    assert matches[0].distance < matches[1].distance < matches[2].distance


def test_a_non_hnsw_query_failure_still_raises_and_does_not_fall_back(monkeypatch):
    """Only the specific, safely-recoverable HNSW filtered-search error
    triggers the brute-force fallback — a genuine failure (e.g. Chroma
    unreachable) must still surface as `ExternalServiceError`, not be
    silently papered over.
    """
    vector_store = _make_vector_store()
    _seed(vector_store, 5)

    def failing_query(**kwargs):
        raise ConnectionError("could not reach chromadb")

    monkeypatch.setattr(vector_store._collection, "query", failing_query)

    with pytest.raises(ExternalServiceError, match="could not reach chromadb"):
        vector_store.query_similar_chunks([1.0, 2.0], top_k=5, where=_WHERE_WORKFLOW)


def test_the_fallback_is_logged_as_a_warning(monkeypatch, caplog):
    vector_store = _make_vector_store()
    _seed(vector_store, 5)

    def failing_query(**kwargs):
        raise RuntimeError("Cannot return the results in a contigious 2D array. Probably ef or M is too small")

    monkeypatch.setattr(vector_store._collection, "query", failing_query)

    with caplog.at_level("WARNING"):
        vector_store.query_similar_chunks([1.0, 2.0], top_k=5, where=_WHERE_WORKFLOW)

    assert any("falling back to exact brute-force similarity search" in message for message in caplog.messages)


# --- The exact reported real-world scale: 8,275 chunks across artifact types ---


def test_the_contact_and_sticket_log_scale_8275_chunks_across_artifact_types():
    """Mirrors the real, reported feature: WORKFLOW/TEST_CASE/ISSUE
    proportioned like the actual 'Contact and Sticket Log' feature
    (1,069 / 6,072 / 1,134 = 8,275 total). Every per-artifact-type query
    must succeed and respect the requested candidate count, regardless
    of whether ChromaDB's own HNSW search or the brute-force fallback
    ultimately answered it.
    """
    vector_store = _make_vector_store()
    counts = {"WORKFLOW": 1069, "TEST_CASE": 6072, "ISSUE": 1134}

    offset = 0
    for artifact_type, count in counts.items():
        ids = [f"chunk-{artifact_type}-{i}" for i in range(count)]
        embeddings = [[float((offset + i) % 41), float(((offset + i) * 7) % 41)] for i in range(count)]
        documents = [f"{artifact_type} chunk body {i}." for i in range(count)]
        metadatas = [
            {
                "chunkId": ids[i],
                "artifactType": artifact_type,
                "feature": "Contact and Sticket Log",
                "documentSource": "source_of_truth",
                "sourceFilename": f"{artifact_type}-{i}.csv",
                "parserName": "CsvParser",
                "parserVersion": "1.0",
            }
            for i in range(count)
        ]
        # `replace_feature_chunks` isn't used here since it deletes the
        # *whole* feature first, which would wipe out the other artifact
        # types already added in this loop — add directly, batched by
        # hand to stay under ChromaDB's own max batch size (5,461).
        for start in range(0, count, 1000):
            end = start + 1000
            vector_store._collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        offset += count

    assert vector_store.count_matching_chunks(where={"feature": "Contact and Sticket Log"}) == 8275

    for artifact_type in counts:
        where = {"$and": [{"feature": "Contact and Sticket Log"}, {"artifactType": artifact_type}]}
        matches = vector_store.query_similar_chunks([1.0, 2.0], top_k=20, where=where)
        assert len(matches) == 20
