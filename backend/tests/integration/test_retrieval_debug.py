"""Integration tests for `GET /api/v1/retrieval/debug` — the read-only,
step-by-step view of the configurable, multi-stage Retrieval Pipeline.
Source of Truth chunks and uploaded document chunks are produced through
the real `/index-feature/{feature}` and `/uploads` -> `/uploads/{id}/embed`
endpoints (not written directly into ChromaDB), so these tests exercise
the same pipeline a real caller would — including the real chunking
engine that assigns `documentId`/`chunkNumber`, which `AdjacentChunkMerger`
depends on.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.dependencies import get_chroma_client, get_history_service, get_openai_client
from app.main import app
from app.services.history_service import HistoryService
from app.services.vector_store_service import VectorStoreService
from tests.fakes import FakeOpenAIClient


def _write_source_of_truth_document(tmp_path, feature: str, folder: str, filename: str, content: str) -> None:
    root = tmp_path / "source_of_truth" / feature / folder
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text(content, encoding="utf-8")


def _long_body(word_count: int = 600) -> str:
    # Exceeds ChunkingEngine's default chunk_size (500 words), so a
    # single section splits into two overlapping, sequential chunks —
    # exactly what AdjacentChunkMerger should recombine.
    return " ".join(f"word{i}" for i in range(word_count))


def _override_retrieval_dependencies(tmp_path) -> tuple[FakeOpenAIClient, chromadb.ClientAPI]:
    fake_openai_client = FakeOpenAIClient()
    chroma_client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    app.dependency_overrides[get_openai_client] = lambda: fake_openai_client
    app.dependency_overrides[get_chroma_client] = lambda: chroma_client
    app.dependency_overrides[get_history_service] = lambda: HistoryService(history_root=tmp_path / "history")
    return fake_openai_client, chroma_client


def _clear_retrieval_dependency_overrides() -> None:
    app.dependency_overrides.pop(get_openai_client, None)
    app.dependency_overrides.pop(get_chroma_client, None)
    app.dependency_overrides.pop(get_history_service, None)


async def _upload_and_embed(client, filename: str = "notes.md", body: str = "Uploaded doc body text.") -> str:
    upload_response = await client.post(
        "/api/v1/uploads", files=[("files", (filename, body.encode("utf-8"), "text/markdown"))]
    )
    assert upload_response.status_code == 200
    session_id = upload_response.json()["uploadSessionId"]

    embed_response = await client.post(f"/api/v1/uploads/{session_id}/embed")
    assert embed_response.status_code == 200
    return session_id


async def test_retrieval_debug_for_a_never_indexed_feature_returns_empty_results(client, tmp_path):
    _override_retrieval_dependencies(tmp_path)

    try:
        response = await client.get(
            "/api/v1/retrieval/debug", params={"query": "how do refunds work?", "feature": "Nonexistent"}
        )

        assert response.status_code == 200
        body = response.json()
        for source_key in ("workflow", "historicalTestCases", "historicalIssues", "uploadedDocuments"):
            source = body[source_key]
            assert source["candidateRetrieved"] == 0
            assert source["candidateChunks"] == []
            assert source["finalReturned"] == 0
            assert source["finalChunks"] == []
            assert source["mergedCount"] == 0
            assert source["mergedChunks"] == []
    finally:
        _clear_retrieval_dependency_overrides()


async def test_retrieval_debug_returns_full_step_by_step_response(client, tmp_path):
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "onboarding.md", "# Role Permissions\n\nWorkflow body text."
    )
    _write_source_of_truth_document(
        tmp_path, "Appointments", "TestCases", "cases.md", "# Login Test\n\nTest case body text."
    )
    _write_source_of_truth_document(
        tmp_path, "Appointments", "IssueSheets", "issues.md", "# Known Bug\n\nIssue body text."
    )
    _override_retrieval_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        session_id = await _upload_and_embed(client)

        response = await client.get(
            "/api/v1/retrieval/debug",
            params={"query": "body text", "feature": "Appointments", "upload_session_id": session_id},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["userQuery"] == "body text"
        assert body["generatedQueryText"] == "body text"
        assert body["queryEmbedding"]["dimension"] > 0
        assert body["queryEmbedding"]["tokenCount"] > 0

        assert body["workflow"]["candidateRequested"] == 20
        assert body["workflow"]["candidateRetrieved"] == 1
        assert body["workflow"]["finalRequested"] == 5
        assert body["workflow"]["finalReturned"] == 1
        assert body["workflow"]["mergedCount"] == 1
        assert body["historicalTestCases"]["candidateRequested"] == 20
        assert body["historicalTestCases"]["mergedCount"] == 1
        assert body["historicalIssues"]["candidateRequested"] == 15
        assert body["historicalIssues"]["mergedCount"] == 1
        assert body["uploadedDocuments"]["candidateRequested"] == 10
        assert body["uploadedDocuments"]["mergedCount"] == 1

        [workflow_chunk] = body["workflow"]["mergedChunks"]
        assert workflow_chunk["artifactType"] == "WORKFLOW"
        assert workflow_chunk["feature"] == "Appointments"
        assert workflow_chunk["sourceFilename"] == "onboarding.md"
        assert workflow_chunk["sectionHeading"] == "Role Permissions"
        assert 0.0 < workflow_chunk["similarityScore"] <= 1.0
        assert workflow_chunk["collectionName"]
        assert "chunkId" in workflow_chunk
        assert "text" in workflow_chunk
        assert "pageNumber" in workflow_chunk
        assert workflow_chunk["documentId"]
        assert workflow_chunk["chunkNumber"] == 1
        assert isinstance(workflow_chunk["vectorDistance"], (int, float))
        # A single, never-merged chunk: no merge diagnostics apply.
        assert workflow_chunk["merged"] is False
        assert workflow_chunk["mergedChunkRange"] is None
        assert workflow_chunk["mergedChunkCount"] is None

        [upload_chunk] = body["uploadedDocuments"]["mergedChunks"]
        assert upload_chunk["sourceFilename"] == "notes.md"
        assert upload_chunk["collectionName"] == f"uploaded_documents_{session_id}"
    finally:
        _clear_retrieval_dependency_overrides()


async def test_retrieval_debug_reports_per_source_similarity_statistics(client, tmp_path):
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "onboarding.md", "# Role Permissions\n\nWorkflow body text."
    )
    _override_retrieval_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        response = await client.get(
            "/api/v1/retrieval/debug", params={"query": "body text", "feature": "Appointments"}
        )

        assert response.status_code == 200
        workflow = response.json()["workflow"]
        [candidate] = workflow["candidateChunks"]
        # Exactly one candidate: average/min/max must all equal its own score.
        assert workflow["averageSimilarity"] == candidate["similarityScore"]
        assert workflow["minimumSimilarity"] == candidate["similarityScore"]
        assert workflow["maximumSimilarity"] == candidate["similarityScore"]

        # A source with nothing retrieved reports zeroed statistics
        # rather than raising or omitting the fields.
        empty_source = response.json()["historicalTestCases"]
        assert empty_source["averageSimilarity"] == 0.0
        assert empty_source["minimumSimilarity"] == 0.0
        assert empty_source["maximumSimilarity"] == 0.0
    finally:
        _clear_retrieval_dependency_overrides()


async def test_retrieval_debug_includes_a_three_stage_pipeline_trace_per_source(client, tmp_path):
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "onboarding.md", "# Role Permissions\n\nWorkflow body text."
    )
    _override_retrieval_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        response = await client.get(
            "/api/v1/retrieval/debug", params={"query": "body text", "feature": "Appointments", "top_k": 5}
        )

        assert response.status_code == 200
        trace = response.json()["workflow"]["pipelineTrace"]
        assert [stage["stage"] for stage in trace] == [
            "Candidate Retrieval",
            "Final Chunk Selection",
            "Adjacent Chunk Merge",
        ]

        candidate_stage, final_stage, merge_stage = trace
        assert candidate_stage["requested"] == 5
        assert candidate_stage["returned"] == 1
        assert 0.0 < candidate_stage["averageSimilarity"] <= 1.0
        assert candidate_stage["minimumSimilarity"] == candidate_stage["maximumSimilarity"]

        assert final_stage["requested"] == 5
        assert final_stage["returned"] == 1
        assert final_stage["averageSimilarity"] is None

        assert merge_stage["requested"] is None
        assert merge_stage["returned"] == 1
        assert merge_stage["averageSimilarity"] is None
    finally:
        _clear_retrieval_dependency_overrides()


async def test_retrieval_debug_merges_adjacent_chunks_from_a_long_document(client, tmp_path):
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "onboarding.md", f"# Role Permissions\n\n{_long_body()}"
    )
    _override_retrieval_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        response = await client.get(
            "/api/v1/retrieval/debug",
            params={"query": "word1 word2", "feature": "Appointments", "top_k": 5},
        )

        assert response.status_code == 200
        workflow = response.json()["workflow"]
        # The long section produced two sequential candidate/final
        # chunks, which Merge Adjacent Chunks recombines into one.
        assert workflow["candidateRetrieved"] == 2
        assert workflow["finalReturned"] == 2
        assert workflow["mergedCount"] == 1
        [merged] = workflow["mergedChunks"]
        assert "word0" in merged["text"]
        assert "word599" in merged["text"]
        assert merged["merged"] is True
        assert merged["mergedChunkRange"] == "1–2"
        assert merged["mergedChunkCount"] == 2

        # The two pre-merge chunks (still visible in finalChunks) were
        # never touched by the merge themselves — only the combined
        # result reports "merged".
        assert all(chunk["merged"] is False for chunk in workflow["finalChunks"])
    finally:
        _clear_retrieval_dependency_overrides()


async def test_retrieval_debug_without_upload_session_id_omits_upload_results(client, tmp_path):
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "onboarding.md", "# Role Permissions\n\nWorkflow body text."
    )
    _override_retrieval_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        response = await client.get(
            "/api/v1/retrieval/debug", params={"query": "body text", "feature": "Appointments"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["uploadedDocuments"]["mergedChunks"] == []
        assert len(body["workflow"]["mergedChunks"]) == 1
    finally:
        _clear_retrieval_dependency_overrides()


async def test_retrieval_debug_respects_top_k_query_param(client, tmp_path):
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "a.md", "# Section A\n\nWorkflow body A."
    )
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "b.md", "# Section B\n\nWorkflow body B, a little longer."
    )
    _override_retrieval_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        response = await client.get(
            "/api/v1/retrieval/debug",
            params={"query": "body text", "feature": "Appointments", "top_k": 1},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["workflow"]["candidateRequested"] == 1
        assert body["workflow"]["finalRequested"] == 1
        assert len(body["workflow"]["mergedChunks"]) == 1
    finally:
        _clear_retrieval_dependency_overrides()


async def test_retrieval_debug_never_calls_openai_chat(client, tmp_path):
    """FakeOpenAIClient only implements `.embeddings.create` (see
    tests/fakes.py) — it has no `.chat` attribute at all, so a successful
    response here is itself proof the endpoint never attempted an OpenAI
    Chat call.
    """
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "onboarding.md", "# Role Permissions\n\nWorkflow body text."
    )
    _override_retrieval_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        response = await client.get(
            "/api/v1/retrieval/debug", params={"query": "body text", "feature": "Appointments"}
        )

        assert response.status_code == 200
    finally:
        _clear_retrieval_dependency_overrides()


async def test_retrieval_debug_does_not_write_source_of_truth_chunks(client, tmp_path):
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "onboarding.md", "# Role Permissions\n\nWorkflow body text."
    )
    _, chroma_client = _override_retrieval_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200
        vector_store = VectorStoreService(client=chroma_client, collection_name="source_of_truth_chunks")
        chunk_count_before = len(vector_store.get_feature_chunks("Appointments"))

        response = await client.get(
            "/api/v1/retrieval/debug", params={"query": "body text", "feature": "Appointments"}
        )
        assert response.status_code == 200

        chunk_count_after = len(vector_store.get_feature_chunks("Appointments"))
        assert chunk_count_after == chunk_count_before
    finally:
        _clear_retrieval_dependency_overrides()


async def test_retrieval_debug_never_creates_a_collection_for_an_unembedded_upload_session(client, tmp_path):
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "onboarding.md", "# Role Permissions\n\nWorkflow body text."
    )
    _, chroma_client = _override_retrieval_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        response = await client.get(
            "/api/v1/retrieval/debug",
            params={
                "query": "body text",
                "feature": "Appointments",
                "upload_session_id": "sess-never-embedded",
            },
        )

        assert response.status_code == 200
        assert response.json()["uploadedDocuments"]["mergedChunks"] == []
        assert VectorStoreService.collection_exists(chroma_client, "uploaded_documents_sess-never-embedded") is False
    finally:
        _clear_retrieval_dependency_overrides()
