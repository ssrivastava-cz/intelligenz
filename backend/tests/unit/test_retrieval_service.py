"""Unit tests for RetrievalService — orchestration only: embed the
query, then for each knowledge source independently run Retrieve
Candidate Chunks (via a retriever) -> Final Chunk Selection ->
Merge Adjacent Chunks. All actual vector search is exercised via a real
`chromadb.EphemeralClient`; only the OpenAI embedding call is faked
(see tests/fakes.py).
"""
import uuid

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.models.retrieval_configuration import RetrievalConfiguration, SourceRetrievalConfig
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store_service import VectorStoreService
from tests.fakes import FakeOpenAIClient

_EMBEDDING_MODEL = "text-embedding-3-small"
_UPLOAD_COLLECTION_PREFIX = "uploaded_documents"
_DEFAULT_TOP_K = 5


def _source_of_truth_metadata(**overrides) -> dict:
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


def _upload_metadata(**overrides) -> dict:
    base = {
        "chunkId": "upload-chunk",
        "artifactType": "USER_UPLOAD",
        "feature": "sess-1",
        "documentSource": "user_upload",
        "sourceFilename": "notes.md",
        "parserName": "MarkdownParser",
        "parserVersion": "1.0",
    }
    base.update(overrides)
    return base


def _unique_session_id() -> str:
    # `chromadb.EphemeralClient()` instances share underlying storage
    # process-wide, so every test uses its own session id to stay
    # isolated regardless of test order.
    return f"sess-{uuid.uuid4().hex[:8]}"


def _make_service(
    fake_client: FakeOpenAIClient | None = None, **kwargs
) -> tuple[RetrievalService, chromadb.ClientAPI, str]:
    client = fake_client if fake_client is not None else FakeOpenAIClient(dimension=2)
    embedding_service = EmbeddingService(client=client, model=_EMBEDDING_MODEL)

    chroma_client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)

    service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        chroma_client=chroma_client,
        upload_collection_prefix=_UPLOAD_COLLECTION_PREFIX,
        default_top_k=_DEFAULT_TOP_K,
        **kwargs,
    )
    return service, chroma_client, collection_name


def test_retrieve_calls_openai_exactly_once_to_embed_the_query():
    fake_client = FakeOpenAIClient(dimension=2)
    service, _, _ = _make_service(fake_client)

    service.retrieve(query_text="ab", feature="Appointments")

    assert fake_client.calls == [["ab"]]


def test_retrieve_reports_query_embedding_dimension_and_token_count():
    fake_client = FakeOpenAIClient(dimension=2)
    service, _, _ = _make_service(fake_client)

    result = service.retrieve(query_text="ab", feature="Appointments")

    assert result.query_embedding_dimension == 2
    assert result.query_embedding_tokens == EmbeddingService(client=fake_client, model=_EMBEDDING_MODEL).count_tokens(
        ["ab"]
    )[0]


def test_retrieve_returns_empty_results_when_nothing_indexed():
    service, _, _ = _make_service()

    result = service.retrieve(query_text="ab", feature="Appointments")

    assert result.workflow_results == []
    assert result.test_case_results == []
    assert result.issue_results == []
    assert result.upload_results == []
    assert result.workflow.candidate_chunks == []
    assert result.workflow.final_chunks == []


def test_retrieve_separates_results_by_artifact_type():
    service, chroma_client, collection_name = _make_service()
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["workflow-chunk", "test-case-chunk", "issue-chunk"],
        embeddings=[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
        documents=["Workflow text.", "Test case text.", "Issue text."],
        metadatas=[
            _source_of_truth_metadata(chunkId="workflow-chunk", artifactType="WORKFLOW"),
            _source_of_truth_metadata(chunkId="test-case-chunk", artifactType="TEST_CASE"),
            _source_of_truth_metadata(chunkId="issue-chunk", artifactType="ISSUE"),
        ],
    )

    result = service.retrieve(query_text="ab", feature="Appointments")

    assert [c.chunk_id for c in result.workflow_results] == ["workflow-chunk"]
    assert [c.chunk_id for c in result.test_case_results] == ["test-case-chunk"]
    assert [c.chunk_id for c in result.issue_results] == ["issue-chunk"]


def test_retrieve_excludes_chunks_from_a_different_feature():
    service, chroma_client, collection_name = _make_service()
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Coding Tool",
        ids=["other-feature-chunk"],
        embeddings=[[2.0, 2.0]],
        documents=["Coding Tool text."],
        metadatas=[_source_of_truth_metadata(chunkId="other-feature-chunk", feature="Coding Tool")],
    )

    result = service.retrieve(query_text="ab", feature="Appointments")

    assert result.workflow_results == []


def test_retrieve_omits_upload_results_when_no_session_id_given():
    service, chroma_client, _ = _make_service()
    session_id = _unique_session_id()
    upload_store = VectorStoreService(
        client=chroma_client, collection_name=f"{_UPLOAD_COLLECTION_PREFIX}_{session_id}"
    )
    upload_store.replace_feature_chunks(
        feature=session_id,
        ids=["upload-chunk"],
        embeddings=[[2.0, 2.0]],
        documents=["Uploaded text."],
        metadatas=[_upload_metadata(feature=session_id)],
    )

    result = service.retrieve(query_text="ab", feature="Appointments")

    assert result.upload_results == []


def test_retrieve_includes_upload_results_when_session_id_given():
    service, chroma_client, _ = _make_service()
    session_id = _unique_session_id()
    upload_store = VectorStoreService(
        client=chroma_client, collection_name=f"{_UPLOAD_COLLECTION_PREFIX}_{session_id}"
    )
    upload_store.replace_feature_chunks(
        feature=session_id,
        ids=["upload-chunk"],
        embeddings=[[2.0, 2.0]],
        documents=["Uploaded text."],
        metadatas=[_upload_metadata(feature=session_id)],
    )

    result = service.retrieve(query_text="ab", feature="Appointments", upload_session_id=session_id)

    assert [c.chunk_id for c in result.upload_results] == ["upload-chunk"]


def test_retrieve_respects_an_explicit_top_k_override():
    service, chroma_client, collection_name = _make_service()
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["a", "b"],
        embeddings=[[2.0, 2.0], [4.0, 4.0]],
        documents=["A.", "B."],
        metadatas=[
            _source_of_truth_metadata(chunkId="a", artifactType="WORKFLOW"),
            _source_of_truth_metadata(chunkId="b", artifactType="WORKFLOW"),
        ],
    )

    result = service.retrieve(query_text="ab", feature="Appointments", top_k=1)

    assert len(result.workflow_results) == 1
    assert result.workflow_results[0].chunk_id == "a"
    # top_k is applied uniformly as both candidate and final count.
    assert result.workflow.candidate_requested == 1
    assert result.workflow.final_requested == 1


def test_retrieve_uses_the_default_configuration_when_nothing_is_overridden():
    service, chroma_client, collection_name = _make_service(
        default_configuration=RetrievalConfiguration(
            workflow=SourceRetrievalConfig(candidate_chunks=20, final_chunks=1),
            historical_test_cases=SourceRetrievalConfig(candidate_chunks=20, final_chunks=8),
            historical_issues=SourceRetrievalConfig(candidate_chunks=15, final_chunks=5),
            uploaded_documents=SourceRetrievalConfig(candidate_chunks=10, final_chunks=3),
        )
    )
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["a", "b"],
        embeddings=[[2.0, 2.0], [4.0, 4.0]],
        documents=["A.", "B."],
        metadatas=[
            _source_of_truth_metadata(chunkId="a", artifactType="WORKFLOW"),
            _source_of_truth_metadata(chunkId="b", artifactType="WORKFLOW"),
        ],
    )

    result = service.retrieve(query_text="ab", feature="Appointments")

    assert result.workflow.candidate_requested == 20
    assert len(result.workflow.candidate_chunks) == 2
    assert result.workflow.final_requested == 1
    assert len(result.workflow.final_chunks) == 1


def test_retrieve_accepts_an_explicit_configuration_with_independent_candidate_and_final_counts():
    service, chroma_client, collection_name = _make_service()
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["a", "b", "c"],
        embeddings=[[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        documents=["A.", "B.", "C."],
        metadatas=[
            _source_of_truth_metadata(chunkId="a", artifactType="WORKFLOW"),
            _source_of_truth_metadata(chunkId="b", artifactType="WORKFLOW"),
            _source_of_truth_metadata(chunkId="c", artifactType="WORKFLOW"),
        ],
    )
    configuration = RetrievalConfiguration(
        workflow=SourceRetrievalConfig(candidate_chunks=3, final_chunks=1),
        historical_test_cases=SourceRetrievalConfig(candidate_chunks=1, final_chunks=1),
        historical_issues=SourceRetrievalConfig(candidate_chunks=1, final_chunks=1),
        uploaded_documents=SourceRetrievalConfig(candidate_chunks=1, final_chunks=1),
    )

    result = service.retrieve(query_text="ab", feature="Appointments", configuration=configuration)

    assert result.workflow.candidate_requested == 3
    assert len(result.workflow.candidate_chunks) == 3
    assert result.workflow.final_requested == 1
    assert len(result.workflow.final_chunks) == 1
    # Final Chunk Selection keeps the best-scoring candidate.
    assert result.workflow.final_chunks[0].chunk_id == "a"


def test_retrieve_reports_candidate_requested_separately_from_candidate_retrieved():
    """Chroma can return fewer matches than requested when a collection
    has fewer eligible chunks — `candidate_requested` must still reflect
    what was asked for, not what came back.
    """
    service, chroma_client, collection_name = _make_service()
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["a"],
        embeddings=[[2.0, 2.0]],
        documents=["A."],
        metadatas=[_source_of_truth_metadata(chunkId="a", artifactType="WORKFLOW")],
    )
    configuration = RetrievalConfiguration(
        workflow=SourceRetrievalConfig(candidate_chunks=20, final_chunks=5),
        historical_test_cases=SourceRetrievalConfig(candidate_chunks=20, final_chunks=5),
        historical_issues=SourceRetrievalConfig(candidate_chunks=20, final_chunks=5),
        uploaded_documents=SourceRetrievalConfig(candidate_chunks=20, final_chunks=5),
    )

    result = service.retrieve(query_text="ab", feature="Appointments", configuration=configuration)

    assert result.workflow.candidate_requested == 20
    assert len(result.workflow.candidate_chunks) == 1


def test_retrieve_merges_adjacent_chunks_from_the_same_document():
    service, chroma_client, collection_name = _make_service()
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["chunk-1", "chunk-2"],
        embeddings=[[2.0, 2.0], [2.0, 2.0]],
        documents=["First half.", "Second half."],
        metadatas=[
            _source_of_truth_metadata(
                chunkId="chunk-1", artifactType="WORKFLOW", documentId="doc-1", chunkNumber=1
            ),
            _source_of_truth_metadata(
                chunkId="chunk-2", artifactType="WORKFLOW", documentId="doc-1", chunkNumber=2
            ),
        ],
    )

    result = service.retrieve(query_text="ab", feature="Appointments", top_k=5)

    assert len(result.workflow.final_chunks) == 2
    assert len(result.workflow.merged_chunks) == 1
    [merged] = result.workflow.merged_chunks
    assert merged.text == "First half.\n\nSecond half."
    assert result.workflow_results == result.workflow.merged_chunks


def test_retrieve_does_not_merge_chunks_from_different_documents():
    service, chroma_client, collection_name = _make_service()
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["chunk-1", "chunk-2"],
        embeddings=[[2.0, 2.0], [2.0, 2.0]],
        documents=["Doc A.", "Doc B."],
        metadatas=[
            _source_of_truth_metadata(
                chunkId="chunk-1", artifactType="WORKFLOW", documentId="doc-a", chunkNumber=1
            ),
            _source_of_truth_metadata(
                chunkId="chunk-2", artifactType="WORKFLOW", documentId="doc-b", chunkNumber=1
            ),
        ],
    )

    result = service.retrieve(query_text="ab", feature="Appointments", top_k=5)

    assert len(result.workflow.merged_chunks) == 2


def test_retrieve_never_merges_chunks_missing_adjacency_metadata():
    """Chunks indexed before documentId/chunkNumber existed have neither
    field — they must never be merged with anything, even other chunks
    from the same feature/collection.
    """
    service, chroma_client, collection_name = _make_service()
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["chunk-1", "chunk-2"],
        embeddings=[[2.0, 2.0], [2.0, 2.0]],
        documents=["First.", "Second."],
        metadatas=[
            _source_of_truth_metadata(chunkId="chunk-1", artifactType="WORKFLOW"),
            _source_of_truth_metadata(chunkId="chunk-2", artifactType="WORKFLOW"),
        ],
    )

    result = service.retrieve(query_text="ab", feature="Appointments", top_k=5)

    assert len(result.workflow.merged_chunks) == 2


# --- retrieve_hybrid: Hybrid Retrieval + Reranking (Knowledge Assistant only) ---
#
# `retrieve()` above is exercised with a fake, non-semantic embedding
# ([len(text)] * dimension) throughout this file — fine for proving
# vector search plumbing, but hybrid retrieval's own ranking signal
# (BM25 + the lexical reranker) is what's under test here, so these
# fixtures use real, readable English text and questions instead.


def test_retrieve_hybrid_returns_empty_results_when_nothing_indexed():
    service, _, _ = _make_service()

    result = service.retrieve_hybrid(query_text="How does Contact Log work?", feature="Contact Log")

    assert result.retrieval_result.workflow_results == []
    assert result.diagnostics.vector_candidate_count == 0
    assert result.diagnostics.bm25_candidate_count == 0
    assert result.diagnostics.final_chunk_count == 0


def test_retrieve_hybrid_finds_a_relevant_chunk_via_vector_and_bm25_together():
    service, chroma_client, collection_name = _make_service(
        hybrid_candidate_chunks=20, reranked_final_chunks=5, reranker_min_score=0.05
    )
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["workflow-chunk"],
        embeddings=[[2.0, 2.0]],
        documents=["Bridged contacts are merged automatically when matching criteria align."],
        metadatas=[
            _source_of_truth_metadata(
                chunkId="workflow-chunk", feature="Contact Log", sectionHeading="Bridged Contacts"
            )
        ],
    )

    result = service.retrieve_hybrid(query_text="How does Contact Log work?", feature="Contact Log")

    assert [c.chunk_id for c in result.retrieval_result.workflow_results] == ["workflow-chunk"]
    [candidate] = result.diagnostics.candidates
    assert candidate.selected is True
    assert candidate.final_rank == 1
    assert candidate.reranker_score > 0


def test_retrieve_hybrid_excludes_a_weakly_related_candidate_below_the_threshold():
    """The reported bug: a generic, topically-adjacent chunk should not
    reach the prompt just because it was retrieved at all."""
    service, chroma_client, collection_name = _make_service(
        hybrid_candidate_chunks=20, reranked_final_chunks=5, reranker_min_score=0.08
    )
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["relevant", "boilerplate"],
        embeddings=[[2.0, 2.0], [2.0, 2.0]],
        documents=[
            "Delete and Undo Delete both apply directly to a Contact Log entry, restoring it within 24 hours.",
            "For more information, browse our knowledge base and other related articles about the product.",
        ],
        metadatas=[
            _source_of_truth_metadata(
                chunkId="relevant", feature="Contact Log", sectionHeading="Delete and Undo Delete"
            ),
            _source_of_truth_metadata(
                chunkId="boilerplate", feature="Contact Log", sectionHeading="Related Articles"
            ),
        ],
    )

    result = service.retrieve_hybrid(
        query_text="How do I delete and undo delete a Contact Log entry?", feature="Contact Log"
    )

    assert [c.chunk_id for c in result.retrieval_result.workflow_results] == ["relevant"]
    boilerplate_candidate = next(c for c in result.diagnostics.candidates if c.chunk_id == "boilerplate")
    assert boilerplate_candidate.selected is False


def test_retrieve_hybrid_deduplicates_near_identical_content_across_documents():
    service, chroma_client, collection_name = _make_service(
        hybrid_candidate_chunks=20, reranked_final_chunks=5, reranker_min_score=0.0
    )
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    duplicate_text = "Bridged contacts are merged automatically when matching criteria align across records."
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["contact-workflow", "sticket-workflow"],
        embeddings=[[2.0, 2.0], [2.1, 2.1]],
        documents=[duplicate_text, duplicate_text],
        metadatas=[
            _source_of_truth_metadata(
                chunkId="contact-workflow", feature="Contact Log", sourceFilename="Contact_Workflow.txt"
            ),
            _source_of_truth_metadata(
                chunkId="sticket-workflow", feature="Contact Log", sourceFilename="Sticket_Workflow.txt"
            ),
        ],
    )

    result = service.retrieve_hybrid(query_text="How does Contact Log work?", feature="Contact Log")

    assert len(result.retrieval_result.workflow_results) == 1
    assert result.diagnostics.duplicates_removed == 1


def test_retrieve_hybrid_respects_the_final_chunks_cap():
    service, chroma_client, collection_name = _make_service(
        hybrid_candidate_chunks=20, reranked_final_chunks=1, reranker_min_score=0.0
    )
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["a", "b"],
        embeddings=[[2.0, 2.0], [2.0, 2.0]],
        documents=[
            "Contact Log workflow step one about bridging contacts.",
            "Contact Log workflow step two about encounter status.",
        ],
        metadatas=[
            _source_of_truth_metadata(chunkId="a", feature="Contact Log"),
            _source_of_truth_metadata(chunkId="b", feature="Contact Log"),
        ],
    )

    result = service.retrieve_hybrid(query_text="How does Contact Log work?", feature="Contact Log")

    assert len(result.retrieval_result.workflow_results) == 1


def test_retrieve_hybrid_still_merges_adjacent_chunks_from_the_same_document():
    """The existing AdjacentChunkMerger keeps running unchanged, after
    reranking/threshold/dedup/top-N — not a second competing merger."""
    service, chroma_client, collection_name = _make_service(
        hybrid_candidate_chunks=20, reranked_final_chunks=5, reranker_min_score=0.0
    )
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["chunk-1", "chunk-2"],
        embeddings=[[2.0, 2.0], [2.0, 2.0]],
        documents=["Contact Log bridging first half.", "Contact Log bridging second half."],
        metadatas=[
            _source_of_truth_metadata(chunkId="chunk-1", feature="Contact Log", documentId="doc-1", chunkNumber=1),
            _source_of_truth_metadata(chunkId="chunk-2", feature="Contact Log", documentId="doc-1", chunkNumber=2),
        ],
    )

    result = service.retrieve_hybrid(query_text="How does Contact Log bridging work?", feature="Contact Log")

    assert len(result.retrieval_result.workflow.merged_chunks) == 1
    [merged] = result.retrieval_result.workflow.merged_chunks
    assert "first half" in merged.text and "second half" in merged.text


def test_retrieve_hybrid_preserves_source_categories():
    service, chroma_client, collection_name = _make_service(hybrid_candidate_chunks=20, reranker_min_score=0.0)
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["workflow-chunk", "test-case-chunk", "issue-chunk"],
        embeddings=[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
        documents=["Appointments workflow text.", "Appointments test case text.", "Appointments issue text."],
        metadatas=[
            _source_of_truth_metadata(chunkId="workflow-chunk", artifactType="WORKFLOW"),
            _source_of_truth_metadata(chunkId="test-case-chunk", artifactType="TEST_CASE"),
            _source_of_truth_metadata(chunkId="issue-chunk", artifactType="ISSUE"),
        ],
    )

    result = service.retrieve_hybrid(query_text="Appointments", feature="Appointments")

    assert [c.chunk_id for c in result.retrieval_result.workflow_results] == ["workflow-chunk"]
    assert [c.chunk_id for c in result.retrieval_result.test_case_results] == ["test-case-chunk"]
    assert [c.chunk_id for c in result.retrieval_result.issue_results] == ["issue-chunk"]


def test_retrieve_hybrid_only_calls_openai_once_to_embed_the_query():
    fake_client = FakeOpenAIClient(dimension=2)
    service, _, _ = _make_service(fake_client)

    service.retrieve_hybrid(query_text="How does Contact Log work?", feature="Contact Log")

    assert fake_client.calls == [["How does Contact Log work?"]]


def test_retrieve_hybrid_measures_stage_latencies_separately():
    service, chroma_client, collection_name = _make_service(hybrid_candidate_chunks=20, reranker_min_score=0.0)
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=["a"],
        embeddings=[[2.0, 2.0]],
        documents=["Contact Log workflow details."],
        metadatas=[_source_of_truth_metadata(chunkId="a")],
    )

    result = service.retrieve_hybrid(query_text="How does Contact Log work?", feature="Contact Log")

    diagnostics = result.diagnostics
    assert diagnostics.vector_retrieval_ms >= 0
    assert diagnostics.bm25_retrieval_ms >= 0
    assert diagnostics.hybrid_fusion_ms >= 0
    assert diagnostics.reranking_ms >= 0
    assert diagnostics.total_retrieval_ms == pytest.approx(
        diagnostics.vector_retrieval_ms
        + diagnostics.bm25_retrieval_ms
        + diagnostics.hybrid_fusion_ms
        + diagnostics.reranking_ms
    )


def test_retrieve_hybrid_reports_requested_available_and_returned_counts_per_source():
    service, chroma_client, collection_name = _make_service(hybrid_candidate_chunks=20, reranker_min_score=0.0)
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)
    ids = [f"chunk-{i}" for i in range(5)]
    vector_store.replace_feature_chunks(
        feature="Contact Log",
        ids=ids,
        embeddings=[[2.0, 2.0]] * 5,
        documents=[f"Contact Log workflow details {i}." for i in range(5)],
        metadatas=[_source_of_truth_metadata(chunkId=cid, feature="Contact Log") for cid in ids],
    )

    result = service.retrieve_hybrid(query_text="How does Contact Log work?", feature="Contact Log")

    by_artifact_type = {query.artifact_type: query for query in result.diagnostics.vector_queries}
    assert set(by_artifact_type) == {"WORKFLOW", "TEST_CASE", "ISSUE"}
    workflow_query = by_artifact_type["WORKFLOW"]
    assert workflow_query.requested_n_results == 20
    assert workflow_query.available_chunks == 5
    assert workflow_query.returned_chunks == 5
    # No test cases or issues were indexed for this feature.
    assert by_artifact_type["TEST_CASE"].available_chunks == 0
    assert by_artifact_type["TEST_CASE"].returned_chunks == 0


def test_retrieve_hybrid_reports_vector_query_diagnostics_for_uploaded_documents():
    service, chroma_client, collection_name = _make_service(hybrid_candidate_chunks=20, reranker_min_score=0.0)
    session_id = _unique_session_id()
    upload_store = VectorStoreService(
        client=chroma_client, collection_name=f"{_UPLOAD_COLLECTION_PREFIX}_{session_id}"
    )
    upload_store.replace_feature_chunks(
        feature=session_id,
        ids=["upload-chunk"],
        embeddings=[[2.0, 2.0]],
        documents=["Uploaded text."],
        metadatas=[_upload_metadata(feature=session_id)],
    )

    result = service.retrieve_hybrid(
        query_text="How does Contact Log work?", feature="Contact Log", upload_session_id=session_id
    )

    by_artifact_type = {query.artifact_type: query for query in result.diagnostics.vector_queries}
    assert "USER_UPLOAD" in by_artifact_type
    assert by_artifact_type["USER_UPLOAD"].available_chunks == 1
    assert by_artifact_type["USER_UPLOAD"].returned_chunks == 1
