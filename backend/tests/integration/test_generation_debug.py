"""Integration tests for `GET /api/v1/generation/debug/{generation_id}` —
purely a read of whatever `HistoryService.save_generation_history`
already persisted. No retrieval, prompt building, or OpenAI call is
involved in this endpoint at all, so setup here calls
`HistoryService.save_generation_history` directly (simulating what the
future Generation Service will do), rather than going through any HTTP
generation endpoint (none exists yet).
"""
from app.core.dependencies import get_history_service
from app.main import app
from app.models.cost import ActualUsage, EstimatedUsage, MoneyAmount, PricingSnapshot
from app.models.generated_test_case import GeneratedTestCase, GeneratedTestStep
from app.models.generation_history import GenerationHistoryDraft, GenerationMetadata, RetrievalSummary
from app.services.history_service import HistoryService


def _make_generated_test_case(**overrides) -> GeneratedTestCase:
    base = {
        "requirement_id": "REQ-1",
        "test_case_id": "TC-1",
        "test_case_title": "Reschedule an appointment",
        "priority": "High",
        "test_suite": "Appointments",
        "preconditions": "User is logged in.",
        "steps": [GeneratedTestStep(step_no=1, action="Open appointment.", expected_result="Details shown.")],
        "post_conditions": "Appointment updated.",
        "automation_status": "Not Automated",
        "test_type": "Functional",
        "tags": ["Appointments"],
    }
    base.update(overrides)
    return GeneratedTestCase(**base)


def _override_history_service(tmp_path) -> HistoryService:
    service = HistoryService(history_root=tmp_path / "history")
    app.dependency_overrides[get_history_service] = lambda: service
    return service


def _clear_history_service_override() -> None:
    app.dependency_overrides.pop(get_history_service, None)


def _make_generation_draft(feature: str = "Appointments", **overrides) -> GenerationHistoryDraft:
    base = {
        "feature": feature,
        "redmine_ticket": "12345",
        "model": "gpt-5",
        "prompt_version": "1.0.0",
        "retrieval": RetrievalSummary(
            workflow_chunks=2, historical_test_cases=1, historical_issues=1, uploaded_documents=1
        ),
        "estimated_usage": EstimatedUsage(
            model="gpt-5",
            input_tokens=1000,
            estimated_input_cost=MoneyAmount(usd=0.00125, inr=0.11),
        ),
        "actual_usage": ActualUsage(
            model="gpt-5",
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
            input_cost=MoneyAmount(usd=0.00125, inr=0.11),
            output_cost=MoneyAmount(usd=0.002, inr=0.17),
            total_cost=MoneyAmount(usd=0.00325, inr=0.28),
        ),
        "pricing": PricingSnapshot(
            model="gpt-5",
            input_price_per_million_tokens=1.25,
            output_price_per_million_tokens=10.00,
            usd_to_inr_exchange_rate=87.00,
        ),
        "generation_time_ms": 2300.0,
        "prompt": "Prompt Version: 1.0.0\n\n## System Instructions\n\n...",
        "response": '{"testCases": [{"testCaseId": "TC-1"}]}',
        "test_cases": [_make_generated_test_case()],
        "metadata": GenerationMetadata(top_k=5, upload_session_id="sess-abc123", generated_test_cases=1),
    }
    base.update(overrides)
    return GenerationHistoryDraft(**base)


async def test_get_generation_debug_returns_the_persisted_entry_exactly(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        saved = history_service.save_generation_history(_make_generation_draft())

        response = await client.get(f"/api/v1/generation/debug/{saved.generation_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["generationId"] == saved.generation_id
        assert body["feature"] == "Appointments"
        assert body["redmineTicket"] == "12345"
        assert body["model"] == "gpt-5"
        assert body["promptVersion"] == "1.0.0"
        assert body["retrieval"] == {
            "workflowChunks": 2,
            "historicalTestCases": 1,
            "historicalIssues": 1,
            "uploadedDocuments": 1,
        }
        assert body["estimatedUsage"] == {
            "model": "gpt-5",
            "inputTokens": 1000,
            "estimatedInputCost": {"usd": 0.00125, "inr": 0.11},
            "currency": "USD",
        }
        assert body["actualUsage"] == {
            "model": "gpt-5",
            "promptTokens": 1000,
            "completionTokens": 200,
            "totalTokens": 1200,
            "inputCost": {"usd": 0.00125, "inr": 0.11},
            "outputCost": {"usd": 0.002, "inr": 0.17},
            "totalCost": {"usd": 0.00325, "inr": 0.28},
        }
        assert body["pricing"] == {
            "model": "gpt-5",
            "inputPricePerMillionTokens": 1.25,
            "outputPricePerMillionTokens": 10.00,
            "usdToInrExchangeRate": 87.00,
        }
        assert body["generationTimeMs"] == 2300.0
        assert body["prompt"] == "Prompt Version: 1.0.0\n\n## System Instructions\n\n..."
        assert body["response"] == '{"testCases": [{"testCaseId": "TC-1"}]}'
        assert len(body["testCases"]) == 1
        [test_case] = body["testCases"]
        assert test_case["testCaseId"] == "TC-1"
        assert test_case["steps"] == [
            {"stepNo": 1, "action": "Open appointment.", "expectedResult": "Details shown."}
        ]
        assert body["metadata"] == {
            "topK": 5,
            "uploadSessionId": "sess-abc123",
            "generatedTestCases": 1,
            # This entry was built directly via `_make_generation_draft`
            # (not through `GenerationService.generate`), so these were
            # never set — same as any generation history persisted
            # before this field existed.
            "requestedTestCases": None,
            "countTargetMet": None,
            "coverageNote": None,
        }
    finally:
        _clear_history_service_override()


async def test_get_generation_debug_404_for_unknown_id(client, tmp_path):
    _override_history_service(tmp_path)

    try:
        response = await client.get("/api/v1/generation/debug/2000-01-01T00-00-00")

        assert response.status_code == 404
    finally:
        _clear_history_service_override()


async def test_get_generation_debug_returns_the_specific_generation_requested(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        first = history_service.save_generation_history(_make_generation_draft(feature="Appointments"))
        second = history_service.save_generation_history(_make_generation_draft(feature="Coding Tool"))

        first_response = await client.get(f"/api/v1/generation/debug/{first.generation_id}")
        second_response = await client.get(f"/api/v1/generation/debug/{second.generation_id}")

        assert first_response.json()["feature"] == "Appointments"
        assert second_response.json()["feature"] == "Coding Tool"
    finally:
        _clear_history_service_override()


async def test_get_generation_debug_does_not_modify_the_stored_history(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        saved = history_service.save_generation_history(_make_generation_draft())

        await client.get(f"/api/v1/generation/debug/{saved.generation_id}")
        await client.get(f"/api/v1/generation/debug/{saved.generation_id}")

        reloaded = history_service.load_generation(saved.generation_id)
        assert reloaded == saved
        assert len(history_service.list_generations()) == 1
    finally:
        _clear_history_service_override()


def test_generation_router_never_imports_retrieval_service_prompt_builder_or_openai():
    """Structural guarantee, not just an absence of failures: the debug
    router can't call RetrievalService, PromptBuilder, or OpenAI even by
    accident, because it never imports them. Checked against the actual
    `import` lines, not the module's prose docstring, which mentions
    these names only to explain what the endpoint deliberately avoids.
    """
    import ast
    import inspect

    from app.routers import generation as generation_router_module

    tree = ast.parse(inspect.getsource(generation_router_module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom | ast.Import)
        for alias in node.names
    }
    for forbidden in ("RetrievalService", "PromptBuilder", "openai", "EmbeddingService"):
        assert forbidden not in imported_names
