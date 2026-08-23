"""Integration tests for `GET /api/v1/prompt/debug` — the read-only view
of the exact prompt that would be sent to the model. Source of Truth and
uploaded document chunks are produced through the real
`/index-feature/{feature}` and `/uploads` -> `/uploads/{id}/embed`
endpoints, exercising the same RetrievalService -> PromptBuilder chain
the endpoint itself uses.
"""
import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.core.dependencies import get_chroma_client, get_cost_calculator, get_history_service, get_openai_client
from app.main import app
from app.services.cost_calculator import CostCalculator
from app.services.history_service import HistoryService
from tests.fakes import FakeOpenAIClient

_TEST_LLM_MODEL = "test-llm-model"
_TEST_INPUT_PRICE_PER_MILLION = 2.0
_TEST_OUTPUT_PRICE_PER_MILLION = 8.0
_TEST_USD_TO_INR_EXCHANGE_RATE = 82.0


def _write_source_of_truth_document(tmp_path, feature: str, folder: str, filename: str, content: str) -> None:
    root = tmp_path / "source_of_truth" / feature / folder
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text(content, encoding="utf-8")


def _override_prompt_dependencies(tmp_path) -> FakeOpenAIClient:
    fake_openai_client = FakeOpenAIClient()
    app.dependency_overrides[get_openai_client] = lambda: fake_openai_client
    app.dependency_overrides[get_chroma_client] = lambda: chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    app.dependency_overrides[get_history_service] = lambda: HistoryService(history_root=tmp_path / "history")
    # Fixed, known-in-advance pricing (distinct from real .env values) so
    # tests can assert exact costs without depending on real config.
    app.dependency_overrides[get_cost_calculator] = lambda: CostCalculator(
        model=_TEST_LLM_MODEL,
        input_price_per_million_tokens=_TEST_INPUT_PRICE_PER_MILLION,
        output_price_per_million_tokens=_TEST_OUTPUT_PRICE_PER_MILLION,
        usd_to_inr_exchange_rate=_TEST_USD_TO_INR_EXCHANGE_RATE,
    )
    return fake_openai_client


def _clear_prompt_dependency_overrides() -> None:
    app.dependency_overrides.pop(get_openai_client, None)
    app.dependency_overrides.pop(get_chroma_client, None)
    app.dependency_overrides.pop(get_cost_calculator, None)
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


async def test_prompt_debug_for_never_indexed_feature_returns_empty_context(client, tmp_path):
    _override_prompt_dependencies(tmp_path)

    try:
        response = await client.get(
            "/api/v1/prompt/debug",
            params={
                "feature": "Nonexistent",
                "redmine_id": "999",
                "redmine_description": "Some description text.",
            },
        )

        assert response.status_code == 200
        body = response.json()
        summary = body["retrievedContextSummary"]
        for source_key in ("workflow", "historicalTestCases", "historicalIssues", "uploadedDocuments"):
            assert summary[source_key]["candidateRetrieved"] == 0
            assert summary[source_key]["finalChunks"] == 0
            assert summary[source_key]["mergedChunks"] == 0
            assert summary[source_key]["averageSimilarity"] == 0.0
        assert summary["totalPromptChunks"] == 0
        assert body["includedChunks"] == []
        # Even with nothing retrieved, the fixed prompt scaffolding
        # (system instructions, generation guidelines, ...) still costs
        # something to send.
        assert body["estimatedUsage"]["inputTokens"] > 0
        assert body["estimatedUsage"]["estimatedInputCost"]["usd"] > 0
        assert body["estimatedUsage"]["estimatedInputCost"]["inr"] > 0
        assert "No workflow sections were retrieved" in body["finalPrompt"]
        assert "Some description text." in body["finalPrompt"]
    finally:
        _clear_prompt_dependency_overrides()


async def test_prompt_debug_estimated_usage_reads_pricing_from_configuration(client, tmp_path):
    """Proves estimated cost is computed from whatever CostCalculator was
    configured with — not a hardcoded price — by overriding it with
    different pricing than `_override_prompt_dependencies`'s default and
    checking the response reflects that different configuration.
    """
    _override_prompt_dependencies(tmp_path)
    app.dependency_overrides[get_cost_calculator] = lambda: CostCalculator(
        model="another-model",
        input_price_per_million_tokens=50.0,
        output_price_per_million_tokens=100.0,
        usd_to_inr_exchange_rate=90.0,
    )

    try:
        response = await client.get(
            "/api/v1/prompt/debug",
            params={"feature": "Appointments", "redmine_id": "1", "redmine_description": "Description text."},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["estimatedUsage"]["model"] == "another-model"
        input_tokens = body["estimatedUsage"]["inputTokens"]
        estimated_input_cost = body["estimatedUsage"]["estimatedInputCost"]
        assert estimated_input_cost["usd"] == pytest.approx((input_tokens / 1_000_000) * 50.0)
        assert estimated_input_cost["inr"] == pytest.approx(estimated_input_cost["usd"] * 90.0, abs=0.01)
    finally:
        _clear_prompt_dependency_overrides()


async def test_prompt_debug_returns_full_response_with_mixed_retrieval(client, tmp_path):
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "onboarding.md", "# Role Permissions\n\nWorkflow body text."
    )
    _write_source_of_truth_document(
        tmp_path, "Appointments", "TestCases", "cases.md", "# Login Test\n\nTest case body text."
    )
    _write_source_of_truth_document(
        tmp_path, "Appointments", "IssueSheets", "issues.md", "# Known Bug\n\nIssue body text."
    )
    _override_prompt_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        session_id = await _upload_and_embed(client)

        response = await client.get(
            "/api/v1/prompt/debug",
            params={
                "feature": "Appointments",
                "redmine_id": "12345",
                "redmine_description": "Users should be able to reschedule appointments.",
                "description": "Focus on the reschedule flow.",
                "session_id": session_id,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["feature"] == "Appointments"
        assert body["redmineTicket"] == "12345"
        assert body["promptVersion"]

        assert body["promptLength"]["characters"] == len(body["finalPrompt"])
        assert body["promptLength"]["words"] == len(body["finalPrompt"].split())
        assert body["promptLength"]["estimatedTokens"] > 0

        summary = body["retrievedContextSummary"]
        assert summary["workflow"]["mergedChunks"] == 1
        assert summary["historicalTestCases"]["mergedChunks"] == 1
        assert summary["historicalIssues"]["mergedChunks"] == 1
        assert summary["uploadedDocuments"]["mergedChunks"] == 1
        assert summary["workflow"]["candidateRetrieved"] == 1
        assert summary["workflow"]["finalChunks"] == 1
        assert summary["totalPromptChunks"] == 4
        for source_key in ("workflow", "historicalTestCases", "historicalIssues", "uploadedDocuments"):
            assert 0.0 < summary[source_key]["averageSimilarity"] <= 1.0

        assert "Users should be able to reschedule appointments." in body["finalPrompt"]
        assert "Focus on the reschedule flow." in body["finalPrompt"]
        assert "Workflow body text." in body["finalPrompt"]
        assert "Test case body text." in body["finalPrompt"]
        assert "Issue body text." in body["finalPrompt"]
        assert "Uploaded doc body text." in body["finalPrompt"]
        assert "Return only valid JSON." in body["finalPrompt"]

        assert len(body["includedChunks"]) == 4
        for chunk in body["includedChunks"]:
            assert set(chunk.keys()) == {
                "chunkId",
                "artifactType",
                "sourceFilename",
                "similarityScore",
                "sectionHeading",
            }

        estimated_usage = body["estimatedUsage"]
        assert estimated_usage["model"] == _TEST_LLM_MODEL
        assert estimated_usage["inputTokens"] == body["promptLength"]["estimatedTokens"]
        expected_usd = (estimated_usage["inputTokens"] / 1_000_000) * _TEST_INPUT_PRICE_PER_MILLION
        assert estimated_usage["estimatedInputCost"]["usd"] == pytest.approx(expected_usd)
        assert estimated_usage["estimatedInputCost"]["inr"] == pytest.approx(
            expected_usd * _TEST_USD_TO_INR_EXCHANGE_RATE, abs=0.01
        )
        assert estimated_usage["currency"] == "USD"

        stats = body["promptStatistics"]
        assert stats["systemInstructionsTokens"] > 0
        assert stats["redmineDescriptionTokens"] > 0
        assert stats["workflowTokens"] > 0
        assert stats["historicalTestCaseTokens"] > 0
        assert stats["historicalIssueTokens"] > 0
        assert stats["uploadedDocumentTokens"] > 0
        assert stats["estimatedTotalPromptTokens"] == body["promptLength"]["estimatedTokens"]
    finally:
        _clear_prompt_dependency_overrides()


async def test_prompt_debug_without_session_id_omits_uploaded_chunks(client, tmp_path):
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "onboarding.md", "# Role Permissions\n\nWorkflow body text."
    )
    _override_prompt_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200

        response = await client.get(
            "/api/v1/prompt/debug",
            params={
                "feature": "Appointments",
                "redmine_id": "12345",
                "redmine_description": "Users should be able to reschedule appointments.",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["retrievedContextSummary"]["uploadedDocuments"]["mergedChunks"] == 0
        assert "No uploaded documents were retrieved" in body["finalPrompt"]
    finally:
        _clear_prompt_dependency_overrides()


async def test_prompt_debug_uses_user_description_as_the_retrieval_query_when_supplied(client, tmp_path):
    fake_client = _override_prompt_dependencies(tmp_path)

    try:
        response = await client.get(
            "/api/v1/prompt/debug",
            params={
                "feature": "Appointments",
                "redmine_id": "12345",
                "redmine_description": "Redmine description text.",
                "description": "User supplied description text.",
            },
        )

        assert response.status_code == 200
        assert fake_client.calls == [["User supplied description text."]]
    finally:
        _clear_prompt_dependency_overrides()


async def test_prompt_debug_falls_back_to_redmine_description_as_the_retrieval_query(client, tmp_path):
    fake_client = _override_prompt_dependencies(tmp_path)

    try:
        response = await client.get(
            "/api/v1/prompt/debug",
            params={
                "feature": "Appointments",
                "redmine_id": "12345",
                "redmine_description": "Redmine description text.",
            },
        )

        assert response.status_code == 200
        assert fake_client.calls == [["Redmine description text."]]
    finally:
        _clear_prompt_dependency_overrides()


async def test_prompt_debug_never_calls_openai_chat(client, tmp_path):
    """FakeOpenAIClient only implements `.embeddings.create` (see
    tests/fakes.py) — it has no `.chat` attribute at all, so a successful
    response here is itself proof the endpoint never attempted an OpenAI
    Chat call.
    """
    _override_prompt_dependencies(tmp_path)

    try:
        response = await client.get(
            "/api/v1/prompt/debug",
            params={
                "feature": "Appointments",
                "redmine_id": "12345",
                "redmine_description": "Redmine description text.",
            },
        )

        assert response.status_code == 200
    finally:
        _clear_prompt_dependency_overrides()


async def test_prompt_debug_does_not_save_generation_or_index_history(client, tmp_path):
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "onboarding.md", "# Role Permissions\n\nWorkflow body text."
    )
    _override_prompt_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200
        history_root = tmp_path / "history"
        index_history_runs_before = list((history_root / "index").iterdir())
        upload_history_before = list(history_root.iterdir())

        response = await client.get(
            "/api/v1/prompt/debug",
            params={
                "feature": "Appointments",
                "redmine_id": "12345",
                "redmine_description": "Redmine description text.",
            },
        )
        assert response.status_code == 200

        index_history_runs_after = list((history_root / "index").iterdir())
        assert index_history_runs_after == index_history_runs_before
        assert not (history_root / "upload").exists() or list((history_root / "upload").iterdir()) == []
        assert list(history_root.iterdir()) == upload_history_before
    finally:
        _clear_prompt_dependency_overrides()
