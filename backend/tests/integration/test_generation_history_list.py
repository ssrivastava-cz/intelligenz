"""Integration tests for `GET /api/v1/generation/history` (paginated
list) and `GET /api/v1/generation/history/stats` (dashboard aggregates)
— both read only persisted generation history via `HistoryService`.
"""
from app.core.dependencies import get_history_service
from app.main import app
from app.models.cost import ActualUsage, EstimatedUsage, MoneyAmount, PricingSnapshot
from app.models.generated_test_case import GeneratedTestCase, GeneratedTestStep
from app.models.generation_history import GenerationHistoryDraft, GenerationMetadata, RetrievalSummary
from app.models.upload_history import UploadHistoryEntry
from app.services.history_service import HistoryService
from app.utils.datetime_utils import utcnow


def _override_history_service(tmp_path) -> HistoryService:
    service = HistoryService(history_root=tmp_path / "history")
    app.dependency_overrides[get_history_service] = lambda: service
    return service


def _clear_history_service_override() -> None:
    app.dependency_overrides.pop(get_history_service, None)


def _make_generated_test_case(**overrides) -> GeneratedTestCase:
    base = {
        "requirement_id": "REQ-1",
        "test_case_id": "TC-1",
        "test_case_title": "Reschedule an appointment",
        "priority": "High",
        "test_suite": "Appointments",
        "preconditions": "User is logged in.",
        "steps": [GeneratedTestStep(step_no=1, action="Open appointment.", expected_result="Details are shown.")],
        "post_conditions": "Appointment is updated.",
        "automation_status": "Not Automated",
        "test_type": "Functional",
        "tags": ["Appointments"],
    }
    base.update(overrides)
    return GeneratedTestCase(**base)


def _make_generation_draft(
    feature: str = "Appointments",
    total_cost_usd: float = 1.0,
    generation_time_ms: float = 1000.0,
    **overrides,
) -> GenerationHistoryDraft:
    base = {
        "feature": feature,
        "redmine_ticket": "12345",
        "model": "gpt-5",
        "prompt_version": "1.0.0",
        "retrieval": RetrievalSummary(
            workflow_chunks=1, historical_test_cases=1, historical_issues=0, uploaded_documents=0
        ),
        "estimated_usage": EstimatedUsage(
            model="gpt-5", input_tokens=1000, estimated_input_cost=MoneyAmount(usd=0.00125, inr=0.11)
        ),
        "actual_usage": ActualUsage(
            model="gpt-5",
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
            input_cost=MoneyAmount(usd=total_cost_usd * 0.5, inr=total_cost_usd * 0.5 * 87),
            output_cost=MoneyAmount(usd=total_cost_usd * 0.5, inr=total_cost_usd * 0.5 * 87),
            total_cost=MoneyAmount(usd=total_cost_usd, inr=total_cost_usd * 87),
        ),
        "pricing": PricingSnapshot(
            model="gpt-5",
            input_price_per_million_tokens=1.25,
            output_price_per_million_tokens=10.00,
            usd_to_inr_exchange_rate=87.00,
        ),
        "generation_time_ms": generation_time_ms,
        "prompt": "Prompt Version: 1.0.0\n\n...",
        "response": '{"testCases": [{"testCaseId": "TC-1"}]}',
        "test_cases": [_make_generated_test_case()],
        "metadata": GenerationMetadata(top_k=5, upload_session_id=None, generated_test_cases=1),
    }
    base.update(overrides)
    return GenerationHistoryDraft(**base)


async def test_list_history_returns_empty_page_when_nothing_saved(client, tmp_path):
    _override_history_service(tmp_path)

    try:
        response = await client.get("/api/v1/generation/history")

        assert response.status_code == 200
        body = response.json()
        assert body == {"items": [], "page": 1, "pageSize": 20, "totalItems": 0, "totalPages": 0}
    finally:
        _clear_history_service_override()


async def test_list_history_returns_summary_fields_only(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        saved = history_service.save_generation_history(_make_generation_draft())

        response = await client.get("/api/v1/generation/history")

        assert response.status_code == 200
        [item] = response.json()["items"]
        assert item == {
            "generationId": saved.generation_id,
            "createdAt": saved.timestamp.isoformat().replace("+00:00", "Z"),
            "feature": "Appointments",
            "redmineTicket": "12345",
            "model": "gpt-5",
            "promptVersion": "1.0.0",
            "generationTimeMs": 1000.0,
            "numberOfTestCases": 1,
            "promptTokens": 1000,
            "completionTokens": 200,
            "totalTokens": 1200,
            "actualGenerationCostUsd": 1.0,
            "actualGenerationCostInr": 87.0,
            "estimatedPromptTokens": 1000,
            "estimatedInputCostUsd": 0.00125,
            "estimatedInputCostInr": 0.11,
            "embeddingTokens": None,
            "embeddingCostUsd": None,
            "embeddingCostInr": None,
            "totalAiCostUsd": 1.0,
            "totalAiCostInr": 87.0,
            "status": "SUCCESS",
        }
    finally:
        _clear_history_service_override()


async def test_list_history_defaults_to_newest_first(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        first = history_service.save_generation_history(_make_generation_draft(feature="First"))
        second = history_service.save_generation_history(_make_generation_draft(feature="Second"))
        third = history_service.save_generation_history(_make_generation_draft(feature="Third"))

        response = await client.get("/api/v1/generation/history")

        ids = [item["generationId"] for item in response.json()["items"]]
        assert ids == [third.generation_id, second.generation_id, first.generation_id]
    finally:
        _clear_history_service_override()


async def test_list_history_sorts_by_requested_field_ascending(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        history_service.save_generation_history(_make_generation_draft(feature="Mid", total_cost_usd=2.0))
        history_service.save_generation_history(_make_generation_draft(feature="Cheap", total_cost_usd=1.0))
        history_service.save_generation_history(_make_generation_draft(feature="Expensive", total_cost_usd=3.0))

        response = await client.get(
            "/api/v1/generation/history", params={"sort": "actualGenerationCostUsd", "order": "asc"}
        )

        features = [item["feature"] for item in response.json()["items"]]
        assert features == ["Cheap", "Mid", "Expensive"]
    finally:
        _clear_history_service_override()


async def test_list_history_sorts_by_requested_field_descending(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        history_service.save_generation_history(_make_generation_draft(feature="Mid", total_cost_usd=2.0))
        history_service.save_generation_history(_make_generation_draft(feature="Cheap", total_cost_usd=1.0))
        history_service.save_generation_history(_make_generation_draft(feature="Expensive", total_cost_usd=3.0))

        response = await client.get(
            "/api/v1/generation/history", params={"sort": "actualGenerationCostUsd", "order": "desc"}
        )

        features = [item["feature"] for item in response.json()["items"]]
        assert features == ["Expensive", "Mid", "Cheap"]
    finally:
        _clear_history_service_override()


async def test_list_history_paginates_results(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        for index in range(5):
            history_service.save_generation_history(_make_generation_draft(feature=f"Feature-{index}"))

        first_page = await client.get(
            "/api/v1/generation/history", params={"page": 1, "page_size": 2, "sort": "feature", "order": "asc"}
        )
        second_page = await client.get(
            "/api/v1/generation/history", params={"page": 2, "page_size": 2, "sort": "feature", "order": "asc"}
        )
        third_page = await client.get(
            "/api/v1/generation/history", params={"page": 3, "page_size": 2, "sort": "feature", "order": "asc"}
        )

        assert [item["feature"] for item in first_page.json()["items"]] == ["Feature-0", "Feature-1"]
        assert [item["feature"] for item in second_page.json()["items"]] == ["Feature-2", "Feature-3"]
        assert [item["feature"] for item in third_page.json()["items"]] == ["Feature-4"]
        assert first_page.json()["totalItems"] == 5
        assert first_page.json()["totalPages"] == 3
    finally:
        _clear_history_service_override()


async def test_list_history_excludes_a_corrupted_file(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        history_service.save_generation_history(_make_generation_draft())
        (tmp_path / "history" / "generation" / "corrupted.json").write_text("{not valid", encoding="utf-8")

        response = await client.get("/api/v1/generation/history")

        assert response.status_code == 200
        assert response.json()["totalItems"] == 1
    finally:
        _clear_history_service_override()


async def test_history_stats_returns_zeroed_values_when_nothing_saved(client, tmp_path):
    _override_history_service(tmp_path)

    try:
        response = await client.get("/api/v1/generation/history/stats")

        assert response.status_code == 200
        body = response.json()
        assert body["totalGenerations"] == 0
        assert body["mostUsedFeature"] is None
        assert body["totalTokens"] == 0
        assert body["totalPromptTokens"] == 0
        assert body["totalCompletionTokens"] == 0
    finally:
        _clear_history_service_override()


async def test_history_stats_computes_aggregates_across_generations(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        history_service.save_generation_history(
            _make_generation_draft(feature="Appointments", total_cost_usd=1.0, generation_time_ms=1000.0)
        )
        history_service.save_generation_history(
            _make_generation_draft(feature="Appointments", total_cost_usd=3.0, generation_time_ms=3000.0)
        )

        response = await client.get("/api/v1/generation/history/stats")

        assert response.status_code == 200
        body = response.json()
        assert body["totalGenerations"] == 2
        assert body["successfulGenerations"] == 2
        assert body["failedGenerations"] == 0
        assert body["averageGenerationTimeMs"] == 2000.0
        assert body["averageCostUsd"] == 2.0
        assert body["totalSpendUsd"] == 4.0
        assert body["mostUsedFeature"] == "Appointments"
        assert body["mostUsedModel"] == "gpt-5"
        # Each draft has prompt_tokens=1000, completion_tokens=200.
        assert body["totalPromptTokens"] == 2000
        assert body["totalCompletionTokens"] == 400
        assert body["totalTokens"] == 2400
    finally:
        _clear_history_service_override()


async def test_list_history_includes_embedding_cost_for_the_generations_upload_session(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        history_service.save_upload_history(
            UploadHistoryEntry(
                upload_session_id="sess-abc123",
                uploaded_at=utcnow(),
                embedding_model="text-embedding-3-small",
                documents_indexed=1,
                chunks_indexed=1,
                embedding_tokens=500,
                average_tokens_per_chunk=500.0,
                estimated_embedding_cost=0.00001,
                elapsed_seconds=0.5,
                chroma_collection_name="uploaded_documents_sess-abc123",
            ),
            [],
        )
        history_service.save_generation_history(
            _make_generation_draft(
                metadata=GenerationMetadata(top_k=5, upload_session_id="sess-abc123", generated_test_cases=1)
            )
        )

        response = await client.get("/api/v1/generation/history")

        [item] = response.json()["items"]
        assert item["embeddingTokens"] == 500
        assert item["embeddingCostUsd"] == 0.00001
        assert item["totalAiCostUsd"] == item["embeddingCostUsd"] + item["actualGenerationCostUsd"]
        assert item["totalAiCostInr"] == item["embeddingCostInr"] + item["actualGenerationCostInr"]
    finally:
        _clear_history_service_override()
