import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.services.vector_store_service import VectorMatch, VectorStoreService


def _make_vector_store(collection_name: str | None = None) -> VectorStoreService:
    # `chromadb.EphemeralClient()` instances share underlying storage for
    # identically-named collections within the same process, so each test
    # gets its own collection name to stay isolated.
    name = collection_name or f"test_collection_{uuid.uuid4().hex[:8]}"
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    return VectorStoreService(client=client, collection_name=name)


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


def test_collection_name_property_reflects_configured_name():
    vector_store = _make_vector_store(collection_name="my_collection")

    assert vector_store.collection_name == "my_collection"


def test_get_feature_chunks_returns_empty_list_when_nothing_stored():
    vector_store = _make_vector_store()

    assert vector_store.get_feature_chunks("Appointments") == []


def test_get_feature_chunks_maps_metadata_and_derives_word_count_and_embedding_fields():
    vector_store = _make_vector_store()
    text = "Role Permissions\n\nUsers with CU role can edit records here today."
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["chunk-1"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        documents=[text],
        metadatas=[_metadata(sectionHeading="Role Permissions")],
    )

    [chunk] = vector_store.get_feature_chunks("Appointments")

    assert chunk.chunk_id == "chunk-1"
    assert chunk.section_heading == "Role Permissions"
    assert chunk.artifact_type == "WORKFLOW"
    assert chunk.source_filename == "onboarding.md"
    assert chunk.word_count == len(text.split())
    assert chunk.embedding_dimension == 4
    assert chunk.embedding_exists is True
    assert chunk.document_source == "source_of_truth"
    assert chunk.page_number is None


def test_get_feature_chunks_reports_embedding_tokens_when_stored():
    vector_store = _make_vector_store()
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["chunk-1"],
        embeddings=[[0.1, 0.2]],
        documents=["Some body text."],
        metadatas=[_metadata(embeddingTokens=42)],
    )

    [chunk] = vector_store.get_feature_chunks("Appointments")

    assert chunk.embedding_tokens == 42


def test_get_feature_chunks_reports_none_embedding_tokens_for_chunks_indexed_before_the_field_existed():
    vector_store = _make_vector_store()
    # No "embeddingTokens" key at all — simulates data written before this
    # field was persisted, not a bug in a later read.
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["chunk-1"],
        embeddings=[[0.1, 0.2]],
        documents=["Some body text."],
        metadatas=[_metadata()],
    )

    [chunk] = vector_store.get_feature_chunks("Appointments")

    assert chunk.embedding_tokens is None


def test_get_feature_chunks_omits_heading_and_page_number_when_not_stored():
    vector_store = _make_vector_store()
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["chunk-1"],
        embeddings=[[0.1, 0.2]],
        documents=["Some body text without a heading."],
        metadatas=[_metadata(artifactType="ISSUE", sourceFilename="issues.txt")],
    )

    [chunk] = vector_store.get_feature_chunks("Appointments")

    assert chunk.section_heading is None
    assert chunk.page_number is None
    assert chunk.artifact_type == "ISSUE"


def test_get_feature_chunks_reports_page_number_when_stored():
    vector_store = _make_vector_store()
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["chunk-1"],
        embeddings=[[0.1, 0.2]],
        documents=["Page one content."],
        metadatas=[_metadata(pageNumber=3)],
    )

    [chunk] = vector_store.get_feature_chunks("Appointments")

    assert chunk.page_number == 3


def test_get_feature_chunks_only_returns_chunks_for_the_requested_feature():
    vector_store = _make_vector_store()
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["a"],
        embeddings=[[0.1]],
        documents=["Appointments body."],
        metadatas=[_metadata(feature="Appointments", chunkId="a", sourceFilename="a.txt")],
    )
    vector_store.replace_feature_chunks(
        feature="Coding Tool",
        ids=["b"],
        embeddings=[[0.2]],
        documents=["Coding Tool body."],
        metadatas=[_metadata(feature="Coding Tool", chunkId="b", sourceFilename="b.txt")],
    )

    chunks = vector_store.get_feature_chunks("Appointments")

    assert [chunk.chunk_id for chunk in chunks] == ["a"]


def test_collection_exists_is_false_for_a_collection_never_created():
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))

    assert VectorStoreService.collection_exists(client, "never-created") is False


def test_collection_exists_does_not_create_a_collection_as_a_side_effect():
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))

    VectorStoreService.collection_exists(client, "never-created")

    assert VectorStoreService.collection_exists(client, "never-created") is False


def test_collection_exists_is_true_once_a_vector_store_has_created_it():
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    name = f"test_collection_{uuid.uuid4().hex[:8]}"
    VectorStoreService(client=client, collection_name=name)

    assert VectorStoreService.collection_exists(client, name) is True


def test_query_similar_chunks_returns_empty_list_when_nothing_stored():
    vector_store = _make_vector_store()

    assert vector_store.query_similar_chunks([0.1, 0.2], top_k=5) == []


def test_query_similar_chunks_returns_empty_list_for_non_positive_top_k():
    vector_store = _make_vector_store()
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        documents=["Some body text."],
        metadatas=[_metadata(chunkId="a")],
    )

    assert vector_store.query_similar_chunks([1.0, 0.0], top_k=0) == []


def test_query_similar_chunks_ranks_by_ascending_distance():
    vector_store = _make_vector_store()
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["near", "far"],
        embeddings=[[1.0, 0.0], [5.0, 5.0]],
        documents=["Near text.", "Far text."],
        metadatas=[_metadata(chunkId="near"), _metadata(chunkId="far")],
    )

    matches = vector_store.query_similar_chunks([1.0, 0.0], top_k=5)

    assert [match.chunk_id for match in matches] == ["near", "far"]
    assert matches[0].distance == 0.0
    assert matches[0].distance < matches[1].distance
    assert isinstance(matches[0], VectorMatch)
    assert matches[0].chunk_text == "Near text."
    assert matches[0].metadata["artifactType"] == "WORKFLOW"


def test_query_similar_chunks_respects_top_k():
    vector_store = _make_vector_store()
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        documents=["A.", "B.", "C."],
        metadatas=[_metadata(chunkId="a"), _metadata(chunkId="b"), _metadata(chunkId="c")],
    )

    matches = vector_store.query_similar_chunks([1.0, 0.0], top_k=1)

    assert len(matches) == 1
    assert matches[0].chunk_id == "a"


def test_query_similar_chunks_applies_a_metadata_where_filter():
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

    matches = vector_store.query_similar_chunks([1.0, 0.0], top_k=5, where={"artifactType": "TEST_CASE"})

    assert [match.chunk_id for match in matches] == ["test-case-chunk"]
