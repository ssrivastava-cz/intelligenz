import json

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.dependencies import get_chroma_client, get_history_service, get_openai_client
from app.main import app
from app.services.history_service import HistoryService
from tests.fakes import FakeOpenAIClient

_VALID_AI_RESPONSE = json.dumps(
    {
        "testCases": [
            {
                "requirementId": "REQ-1",
                "testCaseId": "TC-1",
                "testCaseTitle": "Reschedule an appointment",
                "priority": "High",
                "testSuite": "Appointments",
                "preconditions": "User is logged in.",
                "steps": [{"stepNo": 1, "action": "Open appointment.", "expectedResult": "Details are shown."}],
                "postConditions": "Appointment is updated.",
                "automationStatus": "Not Automated",
                "testType": "Functional",
                "tags": ["Appointments"],
            }
        ]
    }
)


def _write_workflow_document(tmp_path, feature: str) -> None:
    root = tmp_path / "source_of_truth" / feature / "workflows"
    root.mkdir(parents=True)
    (root / "onboarding.md").write_text("# Role Permissions\n\nUsers with CU role can edit records.", encoding="utf-8")


def _override_index_dependencies(tmp_path):
    """`indexer`/`parser_service` already resolve to the tmp_path-scoped
    versions via the `client` fixture's own overrides — only the two
    external boundaries (OpenAI, ChromaDB) and the history file location
    need overriding here.
    """
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


async def test_index_feature_endpoint_indexes_and_records_history(client, tmp_path):
    _write_workflow_document(tmp_path, "Appointments")
    _override_index_dependencies(tmp_path)

    try:
        response = await client.post("/api/v1/index-feature/Appointments")

        assert response.status_code == 200
        body = response.json()
        assert body["feature"] == "Appointments"
        assert body["documentsIndexed"] == 1
        assert body["chunksIndexed"] == 1
        assert body["embeddingModel"] == "text-embedding-3-small"
        assert body["embeddingTokens"] > 0
        assert body["estimatedEmbeddingCost"] >= 0
        assert body["indexedAt"]

        history_response = await client.get("/api/v1/index-history")
        assert history_response.status_code == 200
        history = history_response.json()
        assert len(history) == 1
        assert history[0]["feature"] == "Appointments"
    finally:
        _clear_index_dependency_overrides()


async def test_index_feature_endpoint_404_for_unknown_feature(client, tmp_path):
    _override_index_dependencies(tmp_path)

    try:
        response = await client.post("/api/v1/index-feature/Nonexistent")

        assert response.status_code == 404

        history_response = await client.get("/api/v1/index-history")
        assert history_response.json() == []
    finally:
        _clear_index_dependency_overrides()


async def test_index_source_of_truth_endpoint_indexes_every_feature(client, tmp_path):
    _write_workflow_document(tmp_path, "Appointments")
    _write_workflow_document(tmp_path, "CodingTool")
    _override_index_dependencies(tmp_path)

    try:
        response = await client.post("/api/v1/index-source-of-truth")

        assert response.status_code == 200
        body = response.json()
        assert body["totalFeatures"] == 2
        assert body["successfulFeatures"] == 2
        assert body["failedFeatures"] == 0
        assert body["totalDocumentsIndexed"] == 2
        assert body["totalChunksIndexed"] >= 2
        assert body["totalEmbeddingTokens"] > 0
        assert [f["feature"] for f in body["features"]] == ["Appointments", "CodingTool"]
        assert all(f["status"] == "SUCCESS" and f["error"] is None for f in body["features"])

        history = (await client.get("/api/v1/index-history")).json()
        assert {h["feature"] for h in history} == {"Appointments", "CodingTool"}
    finally:
        _clear_index_dependency_overrides()


async def test_index_source_of_truth_endpoint_reports_a_failed_feature_and_continues(client, tmp_path):
    _write_workflow_document(tmp_path, "Appointments")
    broken = tmp_path / "source_of_truth" / "Broken" / "workflows"
    broken.mkdir(parents=True)
    (broken / "bad.pdf").write_text("this is not a real pdf", encoding="utf-8")
    _override_index_dependencies(tmp_path)

    try:
        response = await client.post("/api/v1/index-source-of-truth")

        assert response.status_code == 200
        body = response.json()
        assert body["totalFeatures"] == 2
        assert body["successfulFeatures"] == 1
        assert body["failedFeatures"] == 1
        by_feature = {f["feature"]: f for f in body["features"]}
        assert by_feature["Broken"]["status"] == "FAILED"
        assert by_feature["Broken"]["error"]
        assert by_feature["Broken"]["chunksIndexed"] == 0
        assert by_feature["Appointments"]["status"] == "SUCCESS"

        # only the successful feature reached the indexing history
        history = (await client.get("/api/v1/index-history")).json()
        assert [h["feature"] for h in history] == ["Appointments"]
    finally:
        _clear_index_dependency_overrides()


async def test_index_source_of_truth_endpoint_returns_an_empty_summary_when_no_features(client, tmp_path):
    (tmp_path / "source_of_truth").mkdir()
    _override_index_dependencies(tmp_path)

    try:
        response = await client.post("/api/v1/index-source-of-truth")

        assert response.status_code == 200
        body = response.json()
        assert body["totalFeatures"] == 0
        assert body["successfulFeatures"] == 0
        assert body["failedFeatures"] == 0
        assert body["features"] == []
        assert (await client.get("/api/v1/index-history")).json() == []
    finally:
        _clear_index_dependency_overrides()


async def test_index_feature_endpoint_is_unchanged_by_the_new_batch_endpoint(client, tmp_path):
    """Regression: POST /index-feature/{feature} still behaves exactly as before."""
    _write_workflow_document(tmp_path, "Appointments")
    _override_index_dependencies(tmp_path)

    try:
        response = await client.post("/api/v1/index-feature/Appointments")

        assert response.status_code == 200
        body = response.json()
        assert body["feature"] == "Appointments"
        assert body["documentsIndexed"] == 1
        assert body["chunksIndexed"] == 1
    finally:
        _clear_index_dependency_overrides()


async def test_index_history_endpoint_returns_empty_list_when_nothing_indexed(client, tmp_path):
    _override_index_dependencies(tmp_path)

    try:
        response = await client.get("/api/v1/index-history")

        assert response.status_code == 200
        assert response.json() == []
    finally:
        _clear_index_dependency_overrides()


# --- GET /index-history/summary and /index-history/stats (Usage Dashboard) ---


async def test_index_history_summary_endpoint_returns_enriched_fields_with_inr_cost(client, tmp_path):
    _write_workflow_document(tmp_path, "Appointments")
    _override_index_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        response = await client.get("/api/v1/index-history/summary")

        assert response.status_code == 200
        [item] = response.json()
        assert item["activityType"] == "DOCUMENT_INDEXING"
        assert item["feature"] == "Appointments"
        assert item["historyId"]
        assert item["embeddingModel"] == "text-embedding-3-small"
        assert item["documentsIndexed"] == 1
        assert item["chunksIndexed"] == 1
        assert item["embeddingTokens"] > 0
        assert item["embeddingCostUsd"] >= 0
        assert item["embeddingCostInr"] >= 0
        assert item["chromaCollectionName"] == "source_of_truth_chunks"
        assert item["status"] == "SUCCESS"
        assert item["indexedAt"]
    finally:
        _clear_index_dependency_overrides()


async def test_index_history_summary_endpoint_returns_empty_list_when_nothing_indexed(client, tmp_path):
    _override_index_dependencies(tmp_path)

    try:
        response = await client.get("/api/v1/index-history/summary")

        assert response.status_code == 200
        assert response.json() == []
    finally:
        _clear_index_dependency_overrides()


async def test_index_history_summary_endpoint_sorts_newest_first(client, tmp_path):
    _write_workflow_document(tmp_path, "Appointments")
    _write_workflow_document(tmp_path, "CodingTool")
    _override_index_dependencies(tmp_path)

    try:
        first = await client.post("/api/v1/index-feature/Appointments")
        second = await client.post("/api/v1/index-feature/CodingTool")
        assert first.status_code == 200
        assert second.status_code == 200

        response = await client.get("/api/v1/index-history/summary")

        features = [item["feature"] for item in response.json()]
        assert features == ["CodingTool", "Appointments"]
    finally:
        _clear_index_dependency_overrides()


async def test_index_history_summary_endpoint_excludes_a_malformed_history_folder(client, tmp_path):
    _write_workflow_document(tmp_path, "Appointments")
    _override_index_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        malformed_folder = tmp_path / "history" / "index" / "2000-01-01_00-00-00"
        malformed_folder.mkdir(parents=True)
        (malformed_folder / "index_summary.json").write_text("{not valid json", encoding="utf-8")

        response = await client.get("/api/v1/index-history/summary")

        assert response.status_code == 200
        [item] = response.json()
        assert item["feature"] == "Appointments"
    finally:
        _clear_index_dependency_overrides()


async def test_index_history_stats_endpoint_returns_zeroed_stats_when_nothing_indexed(client, tmp_path):
    _override_index_dependencies(tmp_path)

    try:
        response = await client.get("/api/v1/index-history/stats")

        assert response.status_code == 200
        body = response.json()
        assert body == {
            "totalIndexingRuns": 0,
            "totalDocumentsIndexed": 0,
            "totalChunksIndexed": 0,
            "totalEmbeddingTokens": 0,
            "totalEmbeddingCostUsd": 0.0,
            "totalEmbeddingCostInr": 0.0,
        }
    finally:
        _clear_index_dependency_overrides()


async def test_index_history_stats_endpoint_aggregates_after_indexing(client, tmp_path):
    _write_workflow_document(tmp_path, "Appointments")
    _write_workflow_document(tmp_path, "Coding Tool")
    _override_index_dependencies(tmp_path)

    try:
        await client.post("/api/v1/index-feature/Appointments")
        await client.post("/api/v1/index-feature/Coding Tool")

        response = await client.get("/api/v1/index-history/stats")

        assert response.status_code == 200
        body = response.json()
        assert body["totalIndexingRuns"] == 2
        assert body["totalDocumentsIndexed"] == 2
        assert body["totalChunksIndexed"] == 2
        assert body["totalEmbeddingTokens"] > 0
        assert body["totalEmbeddingCostUsd"] >= 0
    finally:
        _clear_index_dependency_overrides()


async def test_indexing_cost_is_never_added_to_generation_history_stats(client, tmp_path):
    """Regression test for the reported requirement: indexing and
    generation are different AI activities with different costs — an
    indexing run's embedding cost must never be folded into
    `/generation/history/stats`'s totals, and a generation's cost must
    never be folded into `/index-history/stats`'s totals.
    """
    _write_workflow_document(tmp_path, "Appointments")
    fake_openai_client = FakeOpenAIClient(chat_response_content=_VALID_AI_RESPONSE)
    app.dependency_overrides[get_openai_client] = lambda: fake_openai_client
    app.dependency_overrides[get_chroma_client] = lambda: chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    app.dependency_overrides[get_history_service] = lambda: HistoryService(history_root=tmp_path / "history")

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        generate_response = await client.post("/api/v1/generate", json={"feature": "Appointments"})
        assert generate_response.status_code == 200

        generation_stats = (await client.get("/api/v1/generation/history/stats")).json()
        indexing_stats = (await client.get("/api/v1/index-history/stats")).json()

        assert generation_stats["totalGenerations"] == 1
        assert indexing_stats["totalIndexingRuns"] == 1
        # The generation's own cost is exactly its actual usage cost —
        # never inflated by the indexing run's embedding cost.
        [generation_summary] = (await client.get("/api/v1/generation/history")).json()["items"]
        assert generation_summary["totalAiCostUsd"] == generation_summary["actualGenerationCostUsd"]
        assert generation_stats["totalSpendUsd"] == generation_summary["totalAiCostUsd"]
        # And the indexing run's cost is unaffected by the generation.
        assert indexing_stats["totalEmbeddingCostUsd"] >= 0
    finally:
        app.dependency_overrides.pop(get_openai_client, None)
        app.dependency_overrides.pop(get_chroma_client, None)
        app.dependency_overrides.pop(get_history_service, None)
