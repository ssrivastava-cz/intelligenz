"""Unit tests for `VectorStoreService.replace_feature_chunks`'s write
batching — a feature can produce more chunks than ChromaDB accepts in a
single `collection.add()` call (its own hard limit, `get_max_batch_size()`),
so the write is split into sequential batches. Every test here uses a
real `chromadb.EphemeralClient`, exactly like `tests/unit/test_vector_store_service.py`.
"""
import uuid

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.core.exceptions import ExternalServiceError
from app.services.vector_store_service import VectorStoreService


def _make_vector_store(write_batch_size: int, collection_name: str | None = None) -> VectorStoreService:
    name = collection_name or f"test_collection_{uuid.uuid4().hex[:8]}"
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    return VectorStoreService(client=client, collection_name=name, write_batch_size=write_batch_size)


def _synthetic_chunks(count: int, feature: str = "Contact and Sticket Log"):
    ids = [f"chunk-{i}" for i in range(count)]
    embeddings = [[float(i % 37), float((i * 7) % 37)] for i in range(count)]
    documents = [f"Chunk body text number {i}." for i in range(count)]
    metadatas = [
        {
            "chunkId": ids[i],
            "artifactType": "WORKFLOW",
            "feature": feature,
            "documentSource": "source_of_truth",
            "sourceFilename": f"document-{i}.csv",
            "parserName": "CsvParser",
            "parserVersion": "1.0",
        }
        for i in range(count)
    ]
    return ids, embeddings, documents, metadatas


# --- Batch counts for various chunk counts ---


def test_100_chunks_fit_in_a_single_batch(caplog):
    vector_store = _make_vector_store(write_batch_size=1000)
    ids, embeddings, documents, metadatas = _synthetic_chunks(100)

    with caplog.at_level("INFO"):
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    assert len(vector_store.get_feature_chunks("Contact and Sticket Log")) == 100
    assert any("batch 1/1: 100 chunks" in message for message in caplog.messages)


def test_exactly_batch_size_chunks_fit_in_a_single_batch():
    vector_store = _make_vector_store(write_batch_size=10)
    ids, embeddings, documents, metadatas = _synthetic_chunks(10)

    vector_store.replace_feature_chunks(
        feature="Contact and Sticket Log", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )

    assert len(vector_store.get_feature_chunks("Contact and Sticket Log")) == 10


def test_batch_size_plus_one_chunks_requires_two_batches(caplog):
    vector_store = _make_vector_store(write_batch_size=10)
    ids, embeddings, documents, metadatas = _synthetic_chunks(11)

    with caplog.at_level("INFO"):
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    assert len(vector_store.get_feature_chunks("Contact and Sticket Log")) == 11
    assert any("batch 1/2: 10 chunks" in message for message in caplog.messages)
    assert any("batch 2/2: 1 chunks" in message for message in caplog.messages)


def test_5461_chunks_the_chromadb_maximum_fits_in_one_batch():
    """5461 is ChromaDB's own reported `get_max_batch_size()` for the
    local Chroma version this app uses — configuring a larger write
    batch size must still be clamped down to it, so this many chunks
    fits in exactly one (maximum-sized) batch."""
    vector_store = _make_vector_store(write_batch_size=10_000)
    ids, embeddings, documents, metadatas = _synthetic_chunks(5461)

    vector_store.replace_feature_chunks(
        feature="Contact and Sticket Log", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )

    assert len(vector_store.get_feature_chunks("Contact and Sticket Log")) == 5461


def test_8275_chunks_the_reported_feature_size_writes_successfully(caplog):
    """The exact real-world scenario reported: 'Contact and Sticket Log'
    produces 8,275 chunks — more than ChromaDB's single-batch maximum
    (5,461) — and must be written in full, not truncated."""
    vector_store = _make_vector_store(write_batch_size=1000)
    ids, embeddings, documents, metadatas = _synthetic_chunks(8275)

    with caplog.at_level("INFO"):
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    stored = vector_store.get_feature_chunks("Contact and Sticket Log")
    assert len(stored) == 8275
    assert any("batch 1/9: 1000 chunks" in message for message in caplog.messages)
    assert any("batch 9/9: 275 chunks" in message for message in caplog.messages)
    # Every batch in between is a full 1000-chunk batch.
    for batch_number in range(2, 9):
        assert any(f"batch {batch_number}/9: 1000 chunks" in message for message in caplog.messages)


def test_empty_input_writes_nothing_and_does_not_error():
    vector_store = _make_vector_store(write_batch_size=1000)

    vector_store.replace_feature_chunks(
        feature="Contact and Sticket Log", ids=[], embeddings=[], documents=[], metadatas=[]
    )

    assert vector_store.get_feature_chunks("Contact and Sticket Log") == []


# --- Configurable batch size, clamped to ChromaDB's real maximum ---


def test_configured_batch_size_is_clamped_to_chromadbs_actual_maximum():
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    vector_store = VectorStoreService(
        client=client, collection_name=f"test_{uuid.uuid4().hex[:8]}", write_batch_size=999_999
    )

    assert vector_store._write_batch_size == client.get_max_batch_size()


def test_configured_batch_size_under_the_maximum_is_used_as_is():
    vector_store = _make_vector_store(write_batch_size=250)

    assert vector_store._write_batch_size == 250


# --- Failure during a later batch: no silent success, no partial data left behind ---


def test_a_failure_on_a_later_batch_raises_a_clear_error(monkeypatch):
    vector_store = _make_vector_store(write_batch_size=10)
    ids, embeddings, documents, metadatas = _synthetic_chunks(25)

    original_add = vector_store._collection.add
    call_count = {"n": 0}

    def flaky_add(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated ChromaDB failure")
        return original_add(**kwargs)

    monkeypatch.setattr(vector_store._collection, "add", flaky_add)

    with pytest.raises(ExternalServiceError) as exc_info:
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    message = str(exc_info.value)
    assert "Contact and Sticket Log" in message
    assert "batch 2/3" in message
    assert "10 of 25" in message


def test_a_failure_on_a_later_batch_never_leaves_the_feature_partially_indexed(monkeypatch):
    """Per the required behavior: do not silently report success, and do
    not leave a partial write in place — the feature ends up back at
    zero chunks (a known, re-indexable state) rather than 10/25."""
    vector_store = _make_vector_store(write_batch_size=10)
    ids, embeddings, documents, metadatas = _synthetic_chunks(25)

    original_add = vector_store._collection.add
    call_count = {"n": 0}

    def flaky_add(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated ChromaDB failure")
        return original_add(**kwargs)

    monkeypatch.setattr(vector_store._collection, "add", flaky_add)

    with pytest.raises(ExternalServiceError):
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    assert vector_store.get_feature_chunks("Contact and Sticket Log") == []


def test_a_failure_on_the_first_batch_still_clears_any_stale_existing_chunks(monkeypatch):
    """The pre-write delete already ran before any batch was attempted —
    a failure must not resurrect the *old* index either."""
    vector_store = _make_vector_store(write_batch_size=10)
    ids, embeddings, documents, metadatas = _synthetic_chunks(5)
    vector_store.replace_feature_chunks(
        feature="Contact and Sticket Log", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )
    assert len(vector_store.get_feature_chunks("Contact and Sticket Log")) == 5

    new_ids, new_embeddings, new_documents, new_metadatas = _synthetic_chunks(15)

    def failing_add(**kwargs):
        raise RuntimeError("simulated ChromaDB failure")

    monkeypatch.setattr(vector_store._collection, "add", failing_add)

    with pytest.raises(ExternalServiceError):
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log",
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_documents,
            metadatas=new_metadatas,
        )

    assert vector_store.get_feature_chunks("Contact and Sticket Log") == []


# --- Alignment and duplication across batches ---


def test_ids_embeddings_documents_and_metadata_stay_aligned_across_batches():
    vector_store = _make_vector_store(write_batch_size=10)
    ids, embeddings, documents, metadatas = _synthetic_chunks(35)

    vector_store.replace_feature_chunks(
        feature="Contact and Sticket Log", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )

    stored_by_id = {chunk.chunk_id: chunk for chunk in vector_store.get_feature_chunks("Contact and Sticket Log")}
    assert set(stored_by_id) == set(ids)
    for i, chunk_id in enumerate(ids):
        # Each chunk's own distinct sourceFilename (assigned 1:1 by index
        # in `_synthetic_chunks`) must still match its own id after
        # crossing a batch boundary — proof nothing was shuffled.
        assert stored_by_id[chunk_id].source_filename == f"document-{i}.csv"


def test_no_duplicate_chunks_are_written_across_batches():
    vector_store = _make_vector_store(write_batch_size=10)
    ids, embeddings, documents, metadatas = _synthetic_chunks(35)

    vector_store.replace_feature_chunks(
        feature="Contact and Sticket Log", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )

    stored = vector_store.get_feature_chunks("Contact and Sticket Log")
    stored_ids = [chunk.chunk_id for chunk in stored]
    assert len(stored_ids) == len(set(stored_ids)) == 35


def test_re_indexing_replaces_rather_than_accumulates_across_multiple_batched_writes():
    vector_store = _make_vector_store(write_batch_size=10)
    first_ids, first_embeddings, first_documents, first_metadatas = _synthetic_chunks(25)
    vector_store.replace_feature_chunks(
        feature="Contact and Sticket Log",
        ids=first_ids,
        embeddings=first_embeddings,
        documents=first_documents,
        metadatas=first_metadatas,
    )

    second_ids, second_embeddings, second_documents, second_metadatas = _synthetic_chunks(15)
    vector_store.replace_feature_chunks(
        feature="Contact and Sticket Log",
        ids=second_ids,
        embeddings=second_embeddings,
        documents=second_documents,
        metadatas=second_metadatas,
    )

    stored_ids = {chunk.chunk_id for chunk in vector_store.get_feature_chunks("Contact and Sticket Log")}
    assert stored_ids == set(second_ids)
    assert len(stored_ids) == 15


# --- Mismatched input lengths are rejected before anything is deleted ---


def test_mismatched_input_lengths_are_rejected_before_deleting_the_existing_index():
    vector_store = _make_vector_store(write_batch_size=10)
    ids, embeddings, documents, metadatas = _synthetic_chunks(5)
    vector_store.replace_feature_chunks(
        feature="Contact and Sticket Log", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )

    with pytest.raises(ExternalServiceError):
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log",
            ids=["a", "b"],
            embeddings=[[1.0, 2.0]],  # mismatched length
            documents=["a", "b"],
            metadatas=[{"feature": "Contact and Sticket Log"}, {"feature": "Contact and Sticket Log"}],
        )

    # The existing, valid index from before the bad call must survive.
    assert len(vector_store.get_feature_chunks("Contact and Sticket Log")) == 5


# --- Deletion batching: replace_feature_chunks must delete existing
# chunks by id, in batches, never via collection.delete(where=...) ---


def _seed(
    vector_store: VectorStoreService, count: int, feature: str = "Contact and Sticket Log", prefix: str = "old"
) -> list[str]:
    """Writes `count` chunks with a distinct id prefix (so a test can
    tell "old, pre-existing" chunks apart from the "new" replacement set
    unambiguously) and returns the ids written.
    """
    ids = [f"{prefix}-{i}" for i in range(count)]
    embeddings = [[float(i % 37), float((i * 7) % 37)] for i in range(count)]
    documents = [f"Seed body text number {i}." for i in range(count)]
    metadatas = [
        {
            "chunkId": ids[i],
            "artifactType": "WORKFLOW",
            "feature": feature,
            "documentSource": "source_of_truth",
            "sourceFilename": f"seed-{i}.csv",
            "parserName": "CsvParser",
            "parserVersion": "1.0",
        }
        for i in range(count)
    ]
    vector_store.replace_feature_chunks(
        feature=feature, ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )
    return ids


def test_deleting_zero_existing_chunks_is_a_no_op(caplog):
    vector_store = _make_vector_store(write_batch_size=10)
    new_ids, new_embeddings, new_documents, new_metadatas = _synthetic_chunks(5)

    with caplog.at_level("INFO"):
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log",
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_documents,
            metadatas=new_metadatas,
        )

    assert len(vector_store.get_feature_chunks("Contact and Sticket Log")) == 5
    assert not any("Deleting existing ChromaDB chunks" in message for message in caplog.messages)


def test_deleting_fewer_than_batch_size_existing_chunks(caplog):
    vector_store = _make_vector_store(write_batch_size=10)
    old_ids = _seed(vector_store, 5)
    new_ids, new_embeddings, new_documents, new_metadatas = _synthetic_chunks(3)

    with caplog.at_level("INFO"):
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log",
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_documents,
            metadatas=new_metadatas,
        )

    stored_ids = {chunk.chunk_id for chunk in vector_store.get_feature_chunks("Contact and Sticket Log")}
    assert stored_ids == set(new_ids)
    assert not (stored_ids & set(old_ids))
    assert any("5 chunks in 1 batches" in message for message in caplog.messages)
    assert any("batch 1/1: 5 chunks" in message for message in caplog.messages)


def test_deleting_exactly_batch_size_existing_chunks(caplog):
    vector_store = _make_vector_store(write_batch_size=10)
    old_ids = _seed(vector_store, 10)
    new_ids, new_embeddings, new_documents, new_metadatas = _synthetic_chunks(2)

    with caplog.at_level("INFO"):
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log",
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_documents,
            metadatas=new_metadatas,
        )

    stored_ids = {chunk.chunk_id for chunk in vector_store.get_feature_chunks("Contact and Sticket Log")}
    assert not (stored_ids & set(old_ids))
    assert any("10 chunks in 1 batches" in message for message in caplog.messages)
    assert any("batch 1/1: 10 chunks" in message for message in caplog.messages)


def test_deleting_batch_size_plus_one_existing_chunks_requires_two_batches(caplog):
    vector_store = _make_vector_store(write_batch_size=10)
    old_ids = _seed(vector_store, 11)
    new_ids, new_embeddings, new_documents, new_metadatas = _synthetic_chunks(2)

    with caplog.at_level("INFO"):
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log",
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_documents,
            metadatas=new_metadatas,
        )

    stored_ids = {chunk.chunk_id for chunk in vector_store.get_feature_chunks("Contact and Sticket Log")}
    assert not (stored_ids & set(old_ids))
    assert any("11 chunks in 2 batches" in message for message in caplog.messages)
    assert any("batch 1/2: 10 chunks" in message for message in caplog.messages)
    assert any("batch 2/2: 1 chunks" in message for message in caplog.messages)


def test_deleting_5461_existing_chunks_the_chromadb_maximum(caplog):
    """5461 is ChromaDB's own reported `get_max_batch_size()` — the
    exact count that made `collection.delete(where=...)` fail before
    this fix. A configured batch size above it is still clamped, so
    this many existing chunks deletes in exactly one (maximum-sized)
    batch."""
    vector_store = _make_vector_store(write_batch_size=10_000)
    old_ids = _seed(vector_store, 5461)
    new_ids, new_embeddings, new_documents, new_metadatas = _synthetic_chunks(2)

    with caplog.at_level("INFO"):
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log",
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_documents,
            metadatas=new_metadatas,
        )

    stored_ids = {chunk.chunk_id for chunk in vector_store.get_feature_chunks("Contact and Sticket Log")}
    assert stored_ids == set(new_ids)
    assert not (stored_ids & set(old_ids))
    assert any("5461 chunks in 1 batches" in message for message in caplog.messages)


def test_deleting_8275_existing_chunks_the_reported_feature_size(caplog):
    """The exact real-world scenario reported: re-indexing 'Contact and
    Sticket Log' (8,275 chunks) failed during deletion because
    `collection.delete(where=...)` submitted all 8,275 matching ids at
    once — more than ChromaDB's 5,461 maximum."""
    vector_store = _make_vector_store(write_batch_size=1000)
    old_ids = _seed(vector_store, 8275)
    new_ids, new_embeddings, new_documents, new_metadatas = _synthetic_chunks(2)

    with caplog.at_level("INFO"):
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log",
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_documents,
            metadatas=new_metadatas,
        )

    stored_ids = {chunk.chunk_id for chunk in vector_store.get_feature_chunks("Contact and Sticket Log")}
    assert stored_ids == set(new_ids)
    assert not (stored_ids & set(old_ids))
    assert any("8275 chunks in 9 batches" in message for message in caplog.messages)
    assert any("batch 1/9: 1000 chunks" in message for message in caplog.messages)
    assert any("batch 9/9: 275 chunks" in message for message in caplog.messages)


def test_deletion_batches_preserve_all_ids_none_lost_none_duplicated():
    vector_store = _make_vector_store(write_batch_size=10)
    old_ids = _seed(vector_store, 35)
    assert len(set(old_ids)) == 35  # sanity check the fixture itself has no duplicate ids

    new_ids, new_embeddings, new_documents, new_metadatas = _synthetic_chunks(4)
    vector_store.replace_feature_chunks(
        feature="Contact and Sticket Log",
        ids=new_ids,
        embeddings=new_embeddings,
        documents=new_documents,
        metadatas=new_metadatas,
    )

    stored_ids = {chunk.chunk_id for chunk in vector_store.get_feature_chunks("Contact and Sticket Log")}
    # Every one of the 35 old ids is gone — none left behind, and (since
    # `get_feature_chunks` would surface a duplicate as a repeated id in
    # the list, not just the set) none double-deleted or double-counted.
    assert stored_ids == set(new_ids)


def test_a_failure_during_deletion_stops_the_operation_and_writes_no_new_chunks(monkeypatch):
    """Requirements 6 + 9: a failed delete batch must report which
    batch failed and how many chunks were already deleted, raise
    `ExternalServiceError`, and never proceed to add the new chunks."""
    vector_store = _make_vector_store(write_batch_size=10)
    old_ids = _seed(vector_store, 25)
    new_ids, new_embeddings, new_documents, new_metadatas = _synthetic_chunks(5)

    original_delete = vector_store._collection.delete
    call_count = {"n": 0}

    def flaky_delete(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated ChromaDB delete failure")
        return original_delete(**kwargs)

    monkeypatch.setattr(vector_store._collection, "delete", flaky_delete)

    with pytest.raises(ExternalServiceError) as exc_info:
        vector_store.replace_feature_chunks(
            feature="Contact and Sticket Log",
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_documents,
            metadatas=new_metadatas,
        )

    message = str(exc_info.value)
    assert "Contact and Sticket Log" in message
    assert "batch 2/3" in message
    assert "10 of 25" in message

    stored_ids = {chunk.chunk_id for chunk in vector_store.get_feature_chunks("Contact and Sticket Log")}
    # No new chunks were written at all — deletion failing stopped the
    # operation before the add loop ever started.
    assert not (stored_ids & set(new_ids))
    # The first delete batch (10 chunks) had already succeeded before
    # the second one failed; deletion does not retry or roll that back
    # — only the remaining, not-yet-deleted old chunks are still there.
    assert len(stored_ids) == len(old_ids) - 10
