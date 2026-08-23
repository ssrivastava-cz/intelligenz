"""Integration tests for `GET /api/v1/usage` — the unified Usage
Dashboard view that combines Test Plan Generator history
(`backend/data/history/generation/`, via `HistoryService`) and
Knowledge Assistant usage (`backend/database/generations/*/usage.json`,
via `GenerationRepository`) into one normalized response. Neither
source's own data or behavior is modified by these tests; every test
overrides both dependencies to point at `tmp_path`.
"""
from app.core.dependencies import get_generation_repository, get_history_service
from app.main import app
from app.models.cost import ActualUsage, EstimatedUsage, MoneyAmount, PricingSnapshot
from app.models.generated_test_case import GeneratedTestCase, GeneratedTestStep
from app.models.generation_history import GenerationHistoryDraft, GenerationMetadata, RetrievalSummary
from app.repositories.filesystem_generation_repository import FileSystemGenerationRepository
from app.services.history_service import HistoryService


def _override_usage_dependencies(tmp_path) -> tuple[HistoryService, FileSystemGenerationRepository]:
    history_service = HistoryService(history_root=tmp_path / "history")
    generation_repository = FileSystemGenerationRepository(database_root=tmp_path / "database")
    app.dependency_overrides[get_history_service] = lambda: history_service
    app.dependency_overrides[get_generation_repository] = lambda: generation_repository
    return history_service, generation_repository


def _clear_usage_dependency_overrides() -> None:
    app.dependency_overrides.pop(get_history_service, None)
    app.dependency_overrides.pop(get_generation_repository, None)


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
    feature: str = "Appointments", total_cost_usd: float = 1.0, **overrides
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
        "generation_time_ms": 1000.0,
        "prompt": "Prompt Version: 1.0.0\n\n...",
        "response": '{"testCases": [{"testCaseId": "TC-1"}]}',
        "test_cases": [_make_generated_test_case()],
        "metadata": GenerationMetadata(top_k=5, upload_session_id=None, generated_test_cases=1),
    }
    base.update(overrides)
    return GenerationHistoryDraft(**base)


def _save_knowledge_assistant_generation(
    generation_repository: FileSystemGenerationRepository,
    generation_id: str,
    *,
    model: str = "gpt-5",
    input_tokens: int = 100,
    output_tokens: int = 50,
    total_tokens: int = 150,
    estimated_input_tokens: int = 120,
    input_cost_inr: float = 0.10,
    output_cost_inr: float = 0.20,
    total_cost_inr: float = 0.30,
    generation_time_ms: float = 900.0,
) -> None:
    generation_repository.save_user_question(generation_id, "How does Contact Log work?")
    generation_repository.save_usage(
        generation_id,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_input_tokens=estimated_input_tokens,
        input_cost_inr=input_cost_inr,
        output_cost_inr=output_cost_inr,
        total_cost_inr=total_cost_inr,
        generation_time_ms=generation_time_ms,
    )


async def test_usage_returns_zeroed_summary_and_empty_list_when_both_sources_are_empty(client, tmp_path):
    _override_usage_dependencies(tmp_path)

    try:
        response = await client.get("/api/v1/usage")

        assert response.status_code == 200
        body = response.json()
        assert body["summary"] == {
            "totalGenerations": 0,
            "totalTokens": 0,
            "totalAiCostInr": 0.0,
            "averageCostInr": 0.0,
        }
        assert body["generations"] == []
    finally:
        _clear_usage_dependency_overrides()


async def test_usage_includes_test_plan_generator_records(client, tmp_path):
    history_service, _ = _override_usage_dependencies(tmp_path)

    try:
        saved = history_service.save_generation_history(_make_generation_draft())

        response = await client.get("/api/v1/usage")

        [record] = response.json()["generations"]
        assert record["generationId"] == saved.generation_id
        assert record["source"] == "test_plan_generator"
        assert record["model"] == "gpt-5"
        assert record["inputTokens"] == 1000
        assert record["outputTokens"] == 200
        assert record["totalTokens"] == 1200
        assert record["estimatedInputTokens"] == 1000
        assert record["totalCostInr"] == 87.0
    finally:
        _clear_usage_dependency_overrides()


async def test_usage_returns_empty_generations_when_only_knowledge_assistant_source_is_empty(client, tmp_path):
    """Test Plan Generator data is present, Knowledge Assistant has none
    yet — the dashboard must still show the Test Plan Generator side."""
    history_service, _ = _override_usage_dependencies(tmp_path)

    try:
        history_service.save_generation_history(_make_generation_draft())

        response = await client.get("/api/v1/usage")

        body = response.json()
        assert len(body["generations"]) == 1
        assert body["generations"][0]["source"] == "test_plan_generator"
        assert body["summary"]["totalGenerations"] == 1
    finally:
        _clear_usage_dependency_overrides()


async def test_usage_includes_knowledge_assistant_records_read_directly_from_usage_json(client, tmp_path):
    """Values must come from `usage.json` exactly as persisted — never
    re-derived from `prompt.json`/`response.json` (whose own
    `output_tokens`/`input_tokens` fields may differ, e.g. after a
    retry, and must not leak into the normalized record)."""
    _, generation_repository = _override_usage_dependencies(tmp_path)

    try:
        generation_repository.save_prompt(
            "gen_ka_1", "prompt text", input_tokens=9999, estimated_tokens=9999, prompt_version="v1"
        )
        generation_repository.save_response("gen_ka_1", "answer text", output_tokens=9999)
        _save_knowledge_assistant_generation(
            generation_repository,
            "gen_ka_1",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            estimated_input_tokens=120,
            input_cost_inr=0.10,
            output_cost_inr=0.20,
            total_cost_inr=0.30,
        )

        response = await client.get("/api/v1/usage")

        [record] = response.json()["generations"]
        assert record["generationId"] == "gen_ka_1"
        assert record["source"] == "knowledge_assistant"
        assert record["inputTokens"] == 100
        assert record["outputTokens"] == 50
        assert record["totalTokens"] == 150
        assert record["estimatedInputTokens"] == 120
        assert record["inputCostInr"] == 0.10
        assert record["outputCostInr"] == 0.20
        assert record["totalCostInr"] == 0.30
    finally:
        _clear_usage_dependency_overrides()


async def test_usage_returns_empty_generations_when_only_test_plan_generator_source_is_empty(client, tmp_path):
    _, generation_repository = _override_usage_dependencies(tmp_path)

    try:
        _save_knowledge_assistant_generation(generation_repository, "gen_ka_1")

        response = await client.get("/api/v1/usage")

        body = response.json()
        assert len(body["generations"]) == 1
        assert body["generations"][0]["source"] == "knowledge_assistant"
        assert body["summary"]["totalGenerations"] == 1
    finally:
        _clear_usage_dependency_overrides()


async def test_usage_combines_both_sources_in_one_response(client, tmp_path):
    history_service, generation_repository = _override_usage_dependencies(tmp_path)

    try:
        history_service.save_generation_history(_make_generation_draft())
        _save_knowledge_assistant_generation(generation_repository, "gen_ka_1")

        response = await client.get("/api/v1/usage")

        body = response.json()
        sources = {record["source"] for record in body["generations"]}
        assert sources == {"test_plan_generator", "knowledge_assistant"}
        assert len(body["generations"]) == 2
    finally:
        _clear_usage_dependency_overrides()


async def test_usage_summary_aggregates_generations_and_tokens_across_both_sources(client, tmp_path):
    history_service, generation_repository = _override_usage_dependencies(tmp_path)

    try:
        # Test Plan Generator: 1200 total tokens, total_ai_cost_inr = 87.0.
        history_service.save_generation_history(_make_generation_draft(total_cost_usd=1.0))
        # Knowledge Assistant: 150 total tokens, total_cost_inr = 0.30.
        _save_knowledge_assistant_generation(generation_repository, "gen_ka_1", total_tokens=150, total_cost_inr=0.30)

        response = await client.get("/api/v1/usage")

        summary = response.json()["summary"]
        assert summary["totalGenerations"] == 2
        assert summary["totalTokens"] == 1200 + 150
        assert round(summary["totalAiCostInr"], 2) == round(87.0 + 0.30, 2)
    finally:
        _clear_usage_dependency_overrides()


async def test_usage_average_cost_is_total_cost_divided_by_total_generations_across_both_sources(client, tmp_path):
    history_service, generation_repository = _override_usage_dependencies(tmp_path)

    try:
        history_service.save_generation_history(_make_generation_draft(total_cost_usd=1.0))  # 87.0 INR
        _save_knowledge_assistant_generation(generation_repository, "gen_ka_1", total_cost_inr=13.0)

        response = await client.get("/api/v1/usage")

        summary = response.json()["summary"]
        expected_average = (87.0 + 13.0) / 2
        assert round(summary["averageCostInr"], 4) == round(expected_average, 4)
    finally:
        _clear_usage_dependency_overrides()


async def test_usage_sorts_records_from_both_sources_newest_first(client, tmp_path):
    history_service, generation_repository = _override_usage_dependencies(tmp_path)

    try:
        history_service.save_generation_history(_make_generation_draft(feature="Oldest"))
        _save_knowledge_assistant_generation(generation_repository, "gen_ka_middle")
        history_service.save_generation_history(_make_generation_draft(feature="Newest"))

        response = await client.get("/api/v1/usage")

        created_ats = [record["createdAt"] for record in response.json()["generations"]]
        assert created_ats == sorted(created_ats, reverse=True)
    finally:
        _clear_usage_dependency_overrides()


async def test_usage_skips_a_corrupted_knowledge_assistant_usage_record_without_failing(client, tmp_path):
    history_service, generation_repository = _override_usage_dependencies(tmp_path)

    try:
        history_service.save_generation_history(_make_generation_draft())
        _save_knowledge_assistant_generation(generation_repository, "gen_ka_good")
        corrupted_usage_path = tmp_path / "database" / "generations" / "gen_ka_bad" / "usage.json"
        corrupted_usage_path.parent.mkdir(parents=True)
        corrupted_usage_path.write_text("{not valid json", encoding="utf-8")

        response = await client.get("/api/v1/usage")

        assert response.status_code == 200
        generation_ids = {record["generationId"] for record in response.json()["generations"]}
        assert "gen_ka_bad" not in generation_ids
        assert "gen_ka_good" in generation_ids
    finally:
        _clear_usage_dependency_overrides()


async def test_usage_does_not_change_the_existing_generation_history_endpoint(client, tmp_path):
    """Regression check: adding the unified `/usage` view must not alter
    `GET /generation/history`'s existing response shape or values —
    Test Plan Generator's own history/behavior is read, not modified."""
    history_service, _ = _override_usage_dependencies(tmp_path)

    try:
        saved = history_service.save_generation_history(_make_generation_draft())

        history_response = await client.get("/api/v1/generation/history")

        assert history_response.status_code == 200
        [item] = history_response.json()["items"]
        assert item["generationId"] == saved.generation_id
        assert "source" not in item
    finally:
        _clear_usage_dependency_overrides()
