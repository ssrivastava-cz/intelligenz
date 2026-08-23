"""Unit tests for UploadRetriever — retrieval from an upload session's
dedicated ChromaDB collection, resolved internally from just a session
id, without ever creating a collection as a side effect of a read.
"""
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.retrievers.upload_retriever import UploadRetriever
from app.services.vector_store_service import VectorStoreService

_COLLECTION_PREFIX = "uploaded_documents"


def _metadata(**overrides) -> dict:
    base = {
        "chunkId": "chunk-1",
        "artifactType": "USER_UPLOAD",
        "feature": "sess-1",
        "documentSource": "user_upload",
        "sourceFilename": "notes.md",
        "parserName": "MarkdownParser",
        "parserVersion": "1.0",
    }
    base.update(overrides)
    return base


def _make_client() -> chromadb.ClientAPI:
    return chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))


def _unique_session_id() -> str:
    # `chromadb.EphemeralClient()` instances share underlying storage
    # process-wide, so every test uses its own session id (like
    # tests/unit/test_vector_store_service.py does for collection names)
    # to stay isolated regardless of test order.
    return f"sess-{uuid.uuid4().hex[:8]}"


def test_retrieve_returns_empty_list_for_a_session_never_embedded():
    client = _make_client()
    retriever = UploadRetriever(client, _COLLECTION_PREFIX, upload_session_id=_unique_session_id())

    results = retriever.retrieve([1.0, 0.0], top_k=5)

    assert results == []


def test_retrieve_does_not_create_a_collection_as_a_side_effect():
    client = _make_client()
    session_id = _unique_session_id()
    collection_name = f"{_COLLECTION_PREFIX}_{session_id}"
    retriever = UploadRetriever(client, _COLLECTION_PREFIX, upload_session_id=session_id)

    retriever.retrieve([1.0, 0.0], top_k=5)

    assert VectorStoreService.collection_exists(client, collection_name) is False


def test_retrieve_resolves_the_collection_name_internally():
    client = _make_client()
    session_id = _unique_session_id()
    collection_name = f"{_COLLECTION_PREFIX}_{session_id}"
    vector_store = VectorStoreService(client=client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature=session_id,
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        documents=["Uploaded doc text."],
        metadatas=[_metadata(chunkId="a", feature=session_id)],
    )
    retriever = UploadRetriever(client, _COLLECTION_PREFIX, upload_session_id=session_id)

    [chunk] = retriever.retrieve([1.0, 0.0], top_k=5)

    assert chunk.chunk_id == "a"
    assert chunk.text == "Uploaded doc text."
    assert chunk.collection_name == collection_name
    assert chunk.similarity_score == 1.0


def test_retrieve_only_returns_chunks_from_its_own_sessions_collection():
    client = _make_client()
    session_a = _unique_session_id()
    session_b = _unique_session_id()
    session_a_store = VectorStoreService(client=client, collection_name=f"{_COLLECTION_PREFIX}_{session_a}")
    session_a_store.replace_feature_chunks(
        feature=session_a,
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        documents=["Session A text."],
        metadatas=[_metadata(chunkId="a", feature=session_a)],
    )
    session_b_store = VectorStoreService(client=client, collection_name=f"{_COLLECTION_PREFIX}_{session_b}")
    session_b_store.replace_feature_chunks(
        feature=session_b,
        ids=["b"],
        embeddings=[[1.0, 0.0]],
        documents=["Session B text."],
        metadatas=[_metadata(chunkId="b", feature=session_b)],
    )
    retriever = UploadRetriever(client, _COLLECTION_PREFIX, upload_session_id=session_a)

    results = retriever.retrieve([1.0, 0.0], top_k=5)

    assert [chunk.chunk_id for chunk in results] == ["a"]


def test_retrieve_respects_top_k():
    client = _make_client()
    session_id = _unique_session_id()
    vector_store = VectorStoreService(client=client, collection_name=f"{_COLLECTION_PREFIX}_{session_id}")
    vector_store.replace_feature_chunks(
        feature=session_id,
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [2.0, 0.0]],
        documents=["A.", "B."],
        metadatas=[_metadata(chunkId="a", feature=session_id), _metadata(chunkId="b", feature=session_id)],
    )
    retriever = UploadRetriever(client, _COLLECTION_PREFIX, upload_session_id=session_id)

    results = retriever.retrieve([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "a"
