"""Unit tests for `BM25Retriever` — the lexical (keyword) leg of Hybrid
Retrieval. Real `chromadb.EphemeralClient`, no OpenAI involved at all
(BM25 never calls it) — see `app.retrievers.bm25_retriever`.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.retrievers.bm25_retriever import BM25Retriever
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
    return VectorStoreService(client=client, collection_name="bm25_test")


def test_retrieve_returns_empty_list_when_nothing_indexed():
    vector_store = _vector_store()
    retriever = BM25Retriever(vector_store, feature="Contact Log")

    assert retriever.retrieve("Contact Log", top_k=10) == []


def test_retrieve_finds_a_chunk_containing_exact_terminology():
    """BM25's whole reason for existing: an exact term like "C_10" that
    a bi-encoder embedding can under-rank should be found directly."""
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
    retriever = BM25Retriever(vector_store, feature="Contact Log")

    matches = retriever.retrieve("What does C_10 control?", top_k=10)

    # Only the chunk that actually mentions "C_10" is included at all —
    # the raw BM25 score itself can be small or even slightly negative
    # for a 2-document corpus (a known property of Okapi BM25's IDF
    # term, not a bug — see `BM25Retriever.retrieve`); inclusion is
    # decided by literal term presence, not the score's sign.
    assert [match.chunk.chunk_id for match in matches] == ["chunk-1"]


def test_retrieve_ranks_the_more_lexically_relevant_chunk_first():
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
    retriever = BM25Retriever(vector_store, feature="Contact Log")

    matches = retriever.retrieve("How do I delete and undo delete a Contact Log entry?", top_k=10)

    assert matches[0].chunk.chunk_id == "exact"


def test_retrieve_respects_the_artifact_type_filter():
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
    retriever = BM25Retriever(vector_store, feature="Contact Log")

    matches = retriever.retrieve("Encounter Status", top_k=10, filters={"artifactType": "WORKFLOW"})

    assert [match.chunk.chunk_id for match in matches] == ["workflow-chunk"]


def test_retrieve_excludes_chunks_from_a_different_feature():
    vector_store = _vector_store()
    vector_store.replace_feature_chunks(
        feature="Other Feature",
        ids=["other-chunk"],
        embeddings=[[1.0, 1.0]],
        documents=["Contact Log details that belong to a different feature."],
        metadatas=[_metadata(chunkId="other-chunk", feature="Other Feature")],
    )
    retriever = BM25Retriever(vector_store, feature="Contact Log")

    assert retriever.retrieve("Contact Log", top_k=10) == []


def test_retrieve_returns_no_more_than_top_k():
    vector_store = _vector_store()
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["a", "b", "c"],
        embeddings=[[1.0, 1.0]] * 3,
        documents=["Contact Log detail one.", "Contact Log detail two.", "Contact Log detail three."],
        metadatas=[_metadata(chunkId=cid) for cid in ("a", "b", "c")],
    )
    retriever = BM25Retriever(vector_store, feature="Contact Log")

    matches = retriever.retrieve("Contact Log detail", top_k=2)

    assert len(matches) <= 2


def test_retrieve_returns_empty_list_for_a_query_with_only_stopwords():
    vector_store = _vector_store()
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["chunk-1"],
        embeddings=[[1.0, 1.0]],
        documents=["Contact Log workflow details."],
        metadatas=[_metadata(chunkId="chunk-1")],
    )
    retriever = BM25Retriever(vector_store, feature="Contact Log")

    assert retriever.retrieve("how does the", top_k=10) == []
