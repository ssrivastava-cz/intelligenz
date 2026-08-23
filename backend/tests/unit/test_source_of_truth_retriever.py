"""Unit tests for SourceOfTruthRetriever — retrieval from the single
persistent Source of Truth ChromaDB collection, scoped to one `feature`
at construction and filterable by artifact type per call.
"""
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.retrievers.source_of_truth_retriever import SourceOfTruthRetriever
from app.services.vector_store_service import VectorStoreService


def _metadata(**overrides) -> dict:
    base = {
        "chunkId": "chunk-1",
        "artifactType": "WORKFLOW",
        "feature": "Appointments",
        "documentSource": "source_of_truth",
        "sourceFilename": "onboarding.md",
        "parserName": "MarkdownParser",
        "parserVersion": "1.0",
    }
    base.update(overrides)
    return base


def _make_vector_store() -> VectorStoreService:
    # `chromadb.EphemeralClient()` instances share underlying storage for
    # identically-named collections within the same process, so each test
    # gets its own collection name to stay isolated.
    name = f"test_collection_{uuid.uuid4().hex[:8]}"
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    return VectorStoreService(client=client, collection_name=name)


def test_retrieve_returns_empty_list_when_nothing_indexed():
    vector_store = _make_vector_store()
    retriever = SourceOfTruthRetriever(vector_store, feature="Appointments")

    assert retriever.retrieve([1.0, 0.0], top_k=5) == []


def test_retrieve_scopes_results_to_the_bound_feature():
    vector_store = _make_vector_store()
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        documents=["Appointments text."],
        metadatas=[_metadata(chunkId="a", feature="Appointments")],
    )
    vector_store.replace_feature_chunks(
        feature="Coding Tool",
        ids=["b"],
        embeddings=[[1.0, 0.0]],
        documents=["Coding Tool text."],
        metadatas=[_metadata(chunkId="b", feature="Coding Tool")],
    )
    retriever = SourceOfTruthRetriever(vector_store, feature="Appointments")

    results = retriever.retrieve([1.0, 0.0], top_k=5)

    assert [chunk.chunk_id for chunk in results] == ["a"]
    assert results[0].feature == "Appointments"


def test_retrieve_maps_metadata_into_a_retrieved_chunk():
    vector_store = _make_vector_store()
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        documents=["Role Permissions body text."],
        metadatas=[_metadata(chunkId="a", sectionHeading="Role Permissions", pageNumber=3)],
    )
    retriever = SourceOfTruthRetriever(vector_store, feature="Appointments")

    [chunk] = retriever.retrieve([1.0, 0.0], top_k=5)

    assert chunk.chunk_id == "a"
    assert chunk.text == "Role Permissions body text."
    assert chunk.similarity_score == 1.0  # identical vector -> zero distance
    assert chunk.artifact_type == "WORKFLOW"
    assert chunk.feature == "Appointments"
    assert chunk.source_filename == "onboarding.md"
    assert chunk.section_heading == "Role Permissions"
    assert chunk.page_number == 3
    assert chunk.collection_name == vector_store.collection_name


def test_retrieve_omits_section_heading_and_page_number_when_absent():
    vector_store = _make_vector_store()
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        documents=["Body text without a heading."],
        metadatas=[_metadata(chunkId="a")],
    )
    retriever = SourceOfTruthRetriever(vector_store, feature="Appointments")

    [chunk] = retriever.retrieve([1.0, 0.0], top_k=5)

    assert chunk.section_heading is None
    assert chunk.page_number is None


def test_retrieve_applies_an_additional_artifact_type_filter():
    vector_store = _make_vector_store()
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["workflow-chunk", "test-case-chunk"],
        embeddings=[[1.0, 0.0], [1.0, 0.0]],
        documents=["Workflow text.", "Test case text."],
        metadatas=[
            _metadata(chunkId="workflow-chunk", artifactType="WORKFLOW"),
            _metadata(chunkId="test-case-chunk", artifactType="TEST_CASE"),
        ],
    )
    retriever = SourceOfTruthRetriever(vector_store, feature="Appointments")

    workflow_results = retriever.retrieve([1.0, 0.0], top_k=5, filters={"artifactType": "WORKFLOW"})
    test_case_results = retriever.retrieve([1.0, 0.0], top_k=5, filters={"artifactType": "TEST_CASE"})

    assert [chunk.chunk_id for chunk in workflow_results] == ["workflow-chunk"]
    assert [chunk.chunk_id for chunk in test_case_results] == ["test-case-chunk"]


def test_retrieve_respects_top_k():
    vector_store = _make_vector_store()
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        documents=["A.", "B.", "C."],
        metadatas=[_metadata(chunkId="a"), _metadata(chunkId="b"), _metadata(chunkId="c")],
    )
    retriever = SourceOfTruthRetriever(vector_store, feature="Appointments")

    results = retriever.retrieve([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "a"
