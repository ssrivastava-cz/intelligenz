"""Unit tests for `ephemeral_bm25_search` — the rebuild-per-query lexical
search still used for uploaded documents (per-session ChromaDB
collections that aren't part of the persistent global BM25 index). Real
`chromadb.EphemeralClient`, no OpenAI. This is the algorithm that used to
live in `BM25Retriever`; the assertions are unchanged.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.retrievers.ephemeral_bm25 import ephemeral_bm25_search
from app.services.vector_store_service import VectorStoreService


def _metadata(**overrides) -> dict:
    base = {
        "artifactType": "WORKFLOW",
        "feature": "Contact Log",
        "documentSource": "source_of_truth",
        "sourceFilename": "Contact_Log_Workflow.md",
        "parserName": "MarkdownParser",
        "parserVersion": "1.0",
    }
    base.update(overrides)
    return base


def _vector_store() -> VectorStoreService:
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    return VectorStoreService(client=client, collection_name="ephemeral_bm25_test")


def test_returns_empty_list_when_nothing_indexed():
    assert ephemeral_bm25_search(_vector_store(), "Contact Log", top_k=10, feature="Contact Log") == []


def test_finds_a_chunk_containing_exact_terminology():
    vector_store = _vector_store()
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["chunk-1", "chunk-2"],
        embeddings=[[1.0, 1.0], [1.0, 1.0]],
        documents=[
            "The C_10 field controls encounter status transitions for a contact.",
            "General onboarding steps for new users of the platform.",
        ],
        metadatas=[_metadata(chunkId="chunk-1"), _metadata(chunkId="chunk-2")],
    )

    matches = ephemeral_bm25_search(vector_store, "What does C_10 control?", top_k=10, feature="Contact Log")

    assert [match.chunk.chunk_id for match in matches] == ["chunk-1"]


def test_ranks_the_more_lexically_relevant_chunk_first():
    vector_store = _vector_store()
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["exact", "tangential"],
        embeddings=[[1.0, 1.0], [1.0, 1.0]],
        documents=[
            "Delete and Undo Delete both apply to a Contact Log entry directly.",
            "Contact Log is one of many features available in the product.",
        ],
        metadatas=[_metadata(chunkId="exact"), _metadata(chunkId="tangential")],
    )

    matches = ephemeral_bm25_search(
        vector_store, "How do I delete and undo delete a Contact Log entry?", top_k=10, feature="Contact Log"
    )

    assert matches[0].chunk.chunk_id == "exact"


def test_respects_the_artifact_type_filter():
    vector_store = _vector_store()
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["workflow-chunk", "test-case-chunk"],
        embeddings=[[1.0, 1.0], [1.0, 1.0]],
        documents=["Encounter Status workflow details.", "Encounter Status test case steps."],
        metadatas=[
            _metadata(chunkId="workflow-chunk", artifactType="WORKFLOW"),
            _metadata(chunkId="test-case-chunk", artifactType="TEST_CASE"),
        ],
    )

    matches = ephemeral_bm25_search(
        vector_store, "Encounter Status", top_k=10, feature="Contact Log", filters={"artifactType": "WORKFLOW"}
    )

    assert [match.chunk.chunk_id for match in matches] == ["workflow-chunk"]


def test_returns_no_more_than_top_k():
    vector_store = _vector_store()
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["a", "b", "c"],
        embeddings=[[1.0, 1.0]] * 3,
        documents=["Contact Log detail one.", "Contact Log detail two.", "Contact Log detail three."],
        metadatas=[_metadata(chunkId=cid) for cid in ("a", "b", "c")],
    )

    matches = ephemeral_bm25_search(vector_store, "Contact Log detail", top_k=2, feature="Contact Log")

    assert len(matches) <= 2


def test_returns_empty_list_for_a_query_with_only_stopwords():
    vector_store = _vector_store()
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["chunk-1"],
        embeddings=[[1.0, 1.0]],
        documents=["Contact Log workflow details."],
        metadatas=[_metadata(chunkId="chunk-1")],
    )

    assert ephemeral_bm25_search(vector_store, "how does the", top_k=10, feature="Contact Log") == []


def test_exposes_source_attribution_on_the_returned_chunk():
    vector_store = _vector_store()
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["chunk-1"],
        embeddings=[[1.0, 1.0]],
        documents=["The C_10 field controls encounter status transitions for a contact."],
        metadatas=[
            _metadata(
                chunkId="chunk-1",
                documentTitle="Contact Log Workflow Guide",
                sourcePath="source_of_truth/Contact and Sticket Log/workflows/Contact_Log_Workflow.md",
                sourceFolder="Contact and Sticket Log",
            )
        ],
    )

    [match] = ephemeral_bm25_search(vector_store, "C_10", top_k=10, feature="Contact Log")

    assert match.chunk.document_title == "Contact Log Workflow Guide"
    assert match.chunk.source_path == "source_of_truth/Contact and Sticket Log/workflows/Contact_Log_Workflow.md"
    assert match.chunk.source_folder == "Contact and Sticket Log"
