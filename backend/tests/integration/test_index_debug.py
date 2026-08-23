import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.core.dependencies import get_chroma_client, get_history_service, get_openai_client
from app.main import app
from app.services.history_service import HistoryService
from tests.fakes import FakeOpenAIClient


def _write_workflow_document(tmp_path, feature: str) -> None:
    root = tmp_path / "source_of_truth" / feature / "workflows"
    root.mkdir(parents=True)
    (root / "onboarding.md").write_text("# Role Permissions\n\nUsers with CU role can edit records.", encoding="utf-8")


def _override_index_dependencies(tmp_path) -> FakeOpenAIClient:
    fake_openai_client = FakeOpenAIClient()
    app.dependency_overrides[get_openai_client] = lambda: fake_openai_client
    app.dependency_overrides[get_chroma_client] = lambda: chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    app.dependency_overrides[get_history_service] = lambda: HistoryService(
        history_root=tmp_path / "history"
    )
    return fake_openai_client


def _clear_index_dependency_overrides():
    app.dependency_overrides.pop(get_openai_client, None)
    app.dependency_overrides.pop(get_chroma_client, None)
    app.dependency_overrides.pop(get_history_service, None)


async def test_index_debug_endpoint_404_for_never_indexed_feature(client, tmp_path):
    _override_index_dependencies(tmp_path)

    try:
        response = await client.get("/api/v1/index-debug/Nonexistent")

        assert response.status_code == 404
    finally:
        _clear_index_dependency_overrides()


async def test_index_debug_endpoint_returns_full_debug_payload(client, tmp_path):
    _write_workflow_document(tmp_path, "Appointments")
    _override_index_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200
        indexed_body = index_response.json()

        response = await client.get("/api/v1/index-debug/Appointments")

        assert response.status_code == 200
        body = response.json()
        assert body["feature"] == "Appointments"
        assert body["embeddingModel"] == "text-embedding-3-small"
        assert body["documentsIndexed"] == indexed_body["documentsIndexed"] == 1
        assert body["chunksIndexed"] == indexed_body["chunksIndexed"] == 1
        assert body["embeddingTokens"] == indexed_body["embeddingTokens"]
        assert body["estimatedCost"] == indexed_body["estimatedEmbeddingCost"]
        assert body["averageTokensPerChunk"] == body["embeddingTokens"] / body["chunksIndexed"]
        assert body["elapsedTimeSeconds"] == indexed_body["elapsedSeconds"]
        assert body["indexedAt"] == indexed_body["indexedAt"]
        assert body["collection"]

        [chunk] = body["chunks"]
        assert chunk["sectionHeading"] == "Role Permissions"
        assert chunk["artifactType"] == "WORKFLOW"
        assert chunk["sourceFilename"] == "onboarding.md"
        assert chunk["wordCount"] > 0
        assert chunk["embeddingTokens"] == body["embeddingTokens"]  # only one chunk in this feature
        assert chunk["embeddingTokens"] > 0
        assert chunk["embeddingDimension"] > 0
        assert chunk["embeddingExists"] is True
        assert chunk["documentSource"] == "source_of_truth"
        assert chunk["pageNumber"] is None
        assert chunk["chunkId"]
        assert "embedding" not in chunk
    finally:
        _clear_index_dependency_overrides()


async def test_index_debug_endpoint_computes_average_tokens_per_chunk_across_multiple_chunks(client, tmp_path):
    root = tmp_path / "source_of_truth" / "Appointments" / "workflows"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("SHORT\n\nBrief.", encoding="utf-8")
    (root / "b.txt").write_text(
        "LONG\n\nThis section has quite a lot more words in its body than the other file does.",
        encoding="utf-8",
    )
    _override_index_dependencies(tmp_path)

    try:
        await client.post("/api/v1/index-feature/Appointments")

        response = await client.get("/api/v1/index-debug/Appointments")

        assert response.status_code == 200
        body = response.json()
        assert len(body["chunks"]) == body["chunksIndexed"] == 2
        assert sum(chunk["embeddingTokens"] for chunk in body["chunks"]) == body["embeddingTokens"]
        assert body["averageTokensPerChunk"] == pytest.approx(body["embeddingTokens"] / 2)
        token_counts = {chunk["sourceFilename"]: chunk["embeddingTokens"] for chunk in body["chunks"]}
        assert token_counts["a.txt"] < token_counts["b.txt"]
    finally:
        _clear_index_dependency_overrides()


async def test_index_debug_endpoint_does_not_call_openai(client, tmp_path):
    _write_workflow_document(tmp_path, "Appointments")
    fake_client = _override_index_dependencies(tmp_path)

    try:
        await client.post("/api/v1/index-feature/Appointments")
        calls_before = len(fake_client.calls)

        response = await client.get("/api/v1/index-debug/Appointments")

        assert response.status_code == 200
        assert len(fake_client.calls) == calls_before
    finally:
        _clear_index_dependency_overrides()


async def test_index_debug_endpoint_reflects_the_latest_reindex(client, tmp_path):
    _write_workflow_document(tmp_path, "Appointments")
    _override_index_dependencies(tmp_path)

    try:
        await client.post("/api/v1/index-feature/Appointments")
        second_index_response = await client.post("/api/v1/index-feature/Appointments")
        second_indexed_at = second_index_response.json()["indexedAt"]

        response = await client.get("/api/v1/index-debug/Appointments")

        assert response.status_code == 200
        assert response.json()["indexedAt"] == second_indexed_at
    finally:
        _clear_index_dependency_overrides()


async def test_index_debug_endpoint_reports_multiple_documents_and_matching_chunk_count(client, tmp_path):
    root = tmp_path / "source_of_truth" / "Appointments"
    (root / "workflows").mkdir(parents=True)
    (root / "workflows" / "a.txt").write_text("ROLE PERMISSIONS\n\nBody one.", encoding="utf-8")
    (root / "TestCases").mkdir(parents=True)
    (root / "TestCases" / "cases.csv").write_text("id,title\n1,Book appointment\n", encoding="utf-8")
    _override_index_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        indexed_body = index_response.json()

        response = await client.get("/api/v1/index-debug/Appointments")

        assert response.status_code == 200
        body = response.json()
        assert body["documentsIndexed"] == 2
        assert body["chunksIndexed"] == indexed_body["chunksIndexed"] == len(body["chunks"])
    finally:
        _clear_index_dependency_overrides()
