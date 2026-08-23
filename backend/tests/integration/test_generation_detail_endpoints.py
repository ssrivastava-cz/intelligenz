"""Integration tests for the per-generation History detail endpoints:

    GET    /api/v1/generation/{generation_id}
    GET    /api/v1/generation/{generation_id}/prompt
    GET    /api/v1/generation/{generation_id}/response
    GET    /api/v1/generation/{generation_id}/download/excel
    DELETE /api/v1/generation/{generation_id}

All read (or delete) only persisted generation history via
`HistoryService`; none of them touch OpenAI, retrieval, or ChromaDB.
"""
import io

import openpyxl

from app.core.dependencies import get_history_service
from app.main import app
from app.models.cost import ActualUsage, EstimatedUsage, MoneyAmount, PricingSnapshot
from app.models.generated_test_case import GeneratedTestCase, GeneratedTestStep
from app.models.generation_history import GenerationHistoryDraft, GenerationMetadata, RetrievalSummary
from app.services.history_service import HistoryService

_EXCEL_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


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


def _make_generation_draft(test_cases: list[GeneratedTestCase] | None = None, **overrides) -> GenerationHistoryDraft:
    base = {
        "feature": "Appointments",
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
        "test_cases": test_cases if test_cases is not None else [_make_generated_test_case()],
        "metadata": GenerationMetadata(top_k=5, upload_session_id="sess-abc123", generated_test_cases=1),
    }
    base.update(overrides)
    return GenerationHistoryDraft(**base)


# --- GET /generation/{generation_id} ---


async def test_get_generation_returns_the_complete_persisted_generation(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        saved = history_service.save_generation_history(_make_generation_draft())

        response = await client.get(f"/api/v1/generation/{saved.generation_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["generationId"] == saved.generation_id
        assert body["prompt"] == saved.prompt
        assert body["response"] == saved.response
        assert len(body["testCases"]) == 1
        assert body["retrieval"]["workflowChunks"] == 1
        assert body["actualUsage"]["promptTokens"] == 1000
        assert body["pricing"]["model"] == "gpt-5"
        assert body["metadata"]["uploadSessionId"] == "sess-abc123"
    finally:
        _clear_history_service_override()


async def test_get_generation_404_for_unknown_id(client, tmp_path):
    _override_history_service(tmp_path)

    try:
        response = await client.get("/api/v1/generation/2000-01-01T00-00-00")

        assert response.status_code == 404
    finally:
        _clear_history_service_override()


async def test_get_generation_500_for_a_corrupted_file(client, tmp_path):
    _override_history_service(tmp_path)

    try:
        generation_root = tmp_path / "history" / "generation"
        generation_root.mkdir(parents=True)
        (generation_root / "2026-01-01T00-00-00.json").write_text("{not valid json", encoding="utf-8")

        response = await client.get("/api/v1/generation/2026-01-01T00-00-00")

        assert response.status_code == 500
    finally:
        _clear_history_service_override()


# --- GET /generation/{generation_id}/prompt ---


async def test_get_generation_prompt_returns_plain_text(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        saved = history_service.save_generation_history(_make_generation_draft())

        response = await client.get(f"/api/v1/generation/{saved.generation_id}/prompt")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == saved.prompt
    finally:
        _clear_history_service_override()


async def test_get_generation_prompt_404_for_unknown_id(client, tmp_path):
    _override_history_service(tmp_path)

    try:
        response = await client.get("/api/v1/generation/2000-01-01T00-00-00/prompt")

        assert response.status_code == 404
    finally:
        _clear_history_service_override()


# --- GET /generation/{generation_id}/response ---


async def test_get_generation_response_returns_raw_json_exactly_as_stored(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        saved = history_service.save_generation_history(_make_generation_draft())

        response = await client.get(f"/api/v1/generation/{saved.generation_id}/response")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert response.text == saved.response
    finally:
        _clear_history_service_override()


async def test_get_generation_response_404_for_unknown_id(client, tmp_path):
    _override_history_service(tmp_path)

    try:
        response = await client.get("/api/v1/generation/2000-01-01T00-00-00/response")

        assert response.status_code == 404
    finally:
        _clear_history_service_override()


# --- GET /generation/{generation_id}/download/excel ---


async def test_download_excel_returns_a_valid_workbook(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        saved = history_service.save_generation_history(_make_generation_draft())

        response = await client.get(f"/api/v1/generation/{saved.generation_id}/download/excel")

        assert response.status_code == 200
        assert response.headers["content-type"] == _EXCEL_MEDIA_TYPE
        expected_filename = f"ADO_TestCases_{saved.generation_id}.xlsx"
        assert response.headers["content-disposition"] == f'attachment; filename="{expected_filename}"'

        workbook = openpyxl.load_workbook(io.BytesIO(response.content))
        assert workbook.sheetnames == ["Test Cases", "Generation Summary"]
    finally:
        _clear_history_service_override()


async def test_download_excel_and_export_excel_return_identical_content(client, tmp_path):
    """/download/excel and the earlier /export/excel path both reuse
    ExcelExportService — proving they produce identical bytes confirms
    the logic isn't duplicated between them."""
    history_service = _override_history_service(tmp_path)

    try:
        saved = history_service.save_generation_history(_make_generation_draft())

        download_response = await client.get(f"/api/v1/generation/{saved.generation_id}/download/excel")
        export_response = await client.get(f"/api/v1/generation/{saved.generation_id}/export/excel")

        assert download_response.content == export_response.content
    finally:
        _clear_history_service_override()


async def test_download_excel_409_when_no_test_cases(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        saved = history_service.save_generation_history(
            _make_generation_draft(
                test_cases=[], metadata=GenerationMetadata(top_k=5, upload_session_id=None, generated_test_cases=0)
            )
        )

        response = await client.get(f"/api/v1/generation/{saved.generation_id}/download/excel")

        assert response.status_code == 409
    finally:
        _clear_history_service_override()


async def test_download_excel_404_for_unknown_id(client, tmp_path):
    _override_history_service(tmp_path)

    try:
        response = await client.get("/api/v1/generation/2000-01-01T00-00-00/download/excel")

        assert response.status_code == 404
    finally:
        _clear_history_service_override()


# --- DELETE /generation/{generation_id} ---


async def test_delete_generation_returns_204_and_removes_the_history_file(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        saved = history_service.save_generation_history(_make_generation_draft())

        response = await client.delete(f"/api/v1/generation/{saved.generation_id}")

        assert response.status_code == 204
        assert response.content == b""
        assert history_service.list_generations() == []
    finally:
        _clear_history_service_override()


async def test_delete_generation_404_for_unknown_id(client, tmp_path):
    _override_history_service(tmp_path)

    try:
        response = await client.delete("/api/v1/generation/2000-01-01T00-00-00")

        assert response.status_code == 404
    finally:
        _clear_history_service_override()


async def test_delete_generation_does_not_affect_other_generations(client, tmp_path):
    history_service = _override_history_service(tmp_path)

    try:
        keep = history_service.save_generation_history(_make_generation_draft(feature="Keep"))
        remove = history_service.save_generation_history(_make_generation_draft(feature="Remove"))

        response = await client.delete(f"/api/v1/generation/{remove.generation_id}")

        assert response.status_code == 204
        remaining = history_service.list_generations()
        assert [entry.generation_id for entry in remaining] == [keep.generation_id]
    finally:
        _clear_history_service_override()


# --- route ordering regression ---


async def test_generation_history_path_is_not_shadowed_by_the_dynamic_id_route(client, tmp_path):
    """/generation/history and /generation/{generation_id} are both a
    single path segment past /generation/ — this pins down that the
    literal route wins, not the dynamic one treating "history" as an id.
    """
    _override_history_service(tmp_path)

    try:
        response = await client.get("/api/v1/generation/history")

        assert response.status_code == 200
        assert "items" in response.json()
    finally:
        _clear_history_service_override()


async def test_generation_history_stats_path_is_not_shadowed_by_the_dynamic_id_route(client, tmp_path):
    _override_history_service(tmp_path)

    try:
        response = await client.get("/api/v1/generation/history/stats")

        assert response.status_code == 200
        assert "totalGenerations" in response.json()
    finally:
        _clear_history_service_override()
