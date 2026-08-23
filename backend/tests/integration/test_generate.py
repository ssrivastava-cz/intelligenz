"""Integration tests for `POST /api/v1/generate` — the full AI Test
Case Generation pipeline (RetrievalService -> PromptBuilder ->
GenerationService -> OpenAI Chat -> HistoryService). Source of Truth
and uploaded document chunks are produced through the real
`/index-feature/{feature}` and `/uploads` -> `/uploads/{id}/embed`
endpoints; only OpenAI itself is faked (tests/fakes.py).
"""
import asyncio
import json
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.dependencies import get_chroma_client, get_generation_service, get_history_service, get_openai_client
from app.main import app
from app.services.cost_calculator import CostCalculator
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.history_service import HistoryService
from app.services.prompt_builder import PromptBuilder
from app.services.retrieval_service import RetrievalService
from app.services.vector_store_service import VectorStoreService
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


def _write_source_of_truth_document(tmp_path, feature: str, folder: str, filename: str, content: str) -> None:
    root = tmp_path / "source_of_truth" / feature / folder
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text(content, encoding="utf-8")


def _override_generate_dependencies(tmp_path, chat_response_content: str | None = _VALID_AI_RESPONSE, **fake_kwargs):
    fake_openai_client = FakeOpenAIClient(chat_response_content=chat_response_content, **fake_kwargs)
    # One shared instance, not "a new EphemeralClient per dependency
    # resolution": `POST /generate` now runs in a worker thread (so a
    # concurrent `GET /generate/progress/{id}` poll can still be served
    # while it's in flight), and two threads racing to construct their
    # own first-ever EphemeralClient concurrently can lose a SQLite
    # schema-setup race. A single already-constructed instance sidesteps
    # that entirely, with no behavior change for tests that never touch
    # it concurrently in the first place.
    chroma_client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    # Also a single shared instance rather than "one per resolution":
    # `get_generation_service` is `@lru_cache`d over its resolved
    # dependencies (including this one), so two requests only get the
    # *same* `GenerationService` — needed for one request's progress
    # tracking to be visible to another request polling it — if every
    # dependency it's built from is itself stable across resolutions.
    history_service = HistoryService(history_root=tmp_path / "history")
    app.dependency_overrides[get_openai_client] = lambda: fake_openai_client
    app.dependency_overrides[get_chroma_client] = lambda: chroma_client
    app.dependency_overrides[get_history_service] = lambda: history_service
    return fake_openai_client


def _override_generate_dependencies_with_max_test_cases(
    tmp_path, max_test_cases: int, chat_response_content: str
) -> FakeOpenAIClient:
    """Like `_override_generate_dependencies`, but pins
    `MAX_GENERATED_TEST_CASES` explicitly rather than going through the
    real `get_generation_service`/`get_prompt_builder` (which read
    whatever `backend/.env` the environment running this suite happens
    to have — the exact ambient coupling that silently broke
    `test_generate_endpoint_truncates_test_cases_to_the_configured_maximum`
    once `.env` changed). These tests exist specifically to prove a
    *given* configured value behaves correctly, so the value under test
    must never depend on `.env`. Builds a real `GenerationService` from
    scratch (mirroring `get_generation_service`'s own wiring) rather
    than the shared app dependency graph, since overriding it entirely
    bypasses FastAPI resolving its sub-dependencies.
    """
    fake_openai_client = FakeOpenAIClient(chat_response_content=chat_response_content)
    embedding_service = EmbeddingService(client=fake_openai_client, model="text-embedding-3-small")
    chroma_client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    vector_store = VectorStoreService(client=chroma_client, collection_name=f"test_collection_{uuid.uuid4().hex[:8]}")
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        chroma_client=chroma_client,
        upload_collection_prefix="uploaded_documents",
        default_top_k=5,
    )
    cost_calculator = CostCalculator(
        model="gpt-5",
        input_price_per_million_tokens=1.25,
        output_price_per_million_tokens=10.00,
        usd_to_inr_exchange_rate=87.00,
    )
    history_service = HistoryService(history_root=tmp_path / "history")
    generation_service = GenerationService(
        retrieval_service=retrieval_service,
        prompt_builder=PromptBuilder(max_test_cases=max_test_cases),
        embedding_service=embedding_service,
        openai_client=fake_openai_client,
        cost_calculator=cost_calculator,
        history_service=history_service,
        chat_model="gpt-5",
        max_generated_test_cases=max_test_cases,
    )
    app.dependency_overrides[get_generation_service] = lambda: generation_service
    return fake_openai_client


def _clear_generate_dependency_overrides() -> None:
    app.dependency_overrides.pop(get_openai_client, None)
    app.dependency_overrides.pop(get_chroma_client, None)
    app.dependency_overrides.pop(get_history_service, None)
    app.dependency_overrides.pop(get_generation_service, None)


async def _upload_and_embed(client, filename: str = "notes.md", body: str = "Uploaded doc body text.") -> str:
    upload_response = await client.post(
        "/api/v1/uploads", files=[("files", (filename, body.encode("utf-8"), "text/markdown"))]
    )
    assert upload_response.status_code == 200
    session_id = upload_response.json()["uploadSessionId"]

    embed_response = await client.post(f"/api/v1/uploads/{session_id}/embed")
    assert embed_response.status_code == 200
    return session_id


async def test_generate_endpoint_returns_full_response_on_success(client, tmp_path):
    _override_generate_dependencies(tmp_path)

    try:
        response = await client.post(
            "/api/v1/generate",
            json={
                "feature": "Appointments",
                "redmineId": "12345",
                "redmineDescription": "Users should be able to reschedule appointments.",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["generationId"]
        assert body["model"]
        assert body["promptVersion"]
        assert body["generatedTestCases"] == 1
        assert body["generationTimeMs"] > 0

        actual_usage = body["actualUsage"]
        assert actual_usage["promptTokens"] > 0
        assert actual_usage["completionTokens"] > 0
        assert actual_usage["totalTokens"] == actual_usage["promptTokens"] + actual_usage["completionTokens"]
        assert set(actual_usage["inputCost"].keys()) == {"usd", "inr"}
        assert set(actual_usage["outputCost"].keys()) == {"usd", "inr"}
        assert set(actual_usage["totalCost"].keys()) == {"usd", "inr"}

        [test_case] = body["testCases"]
        assert test_case["testCaseId"] == "TC-1"
        assert test_case["steps"] == [
            {"stepNo": 1, "action": "Open appointment.", "expectedResult": "Details are shown."}
        ]
        # ADO fields: never trusted from the model — always applied by
        # `GenerationService._apply_ado_fields`. `_VALID_AI_RESPONSE`
        # supplies no `testClassification`, so it defaults to "Regression".
        assert test_case["tags"] == ["Appointments", "Regression"]
        assert test_case["workItemType"] == "Test Case"
        assert test_case["automationStatus"] == "Not Automated"
        assert test_case["state"] == "Design"
        assert test_case["areaPath"] == ""
        assert test_case["assignedTo"] == ""
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_persists_history_retrievable_via_debug_endpoint(client, tmp_path):
    _write_source_of_truth_document(
        tmp_path, "Appointments", "workflows", "onboarding.md", "# Role Permissions\n\nWorkflow body text."
    )
    _override_generate_dependencies(tmp_path)

    try:
        index_response = await client.post("/api/v1/index-feature/Appointments")
        assert index_response.status_code == 200
        session_id = await _upload_and_embed(client)

        generate_response = await client.post(
            "/api/v1/generate",
            json={
                "feature": "Appointments",
                "redmineId": "12345",
                "redmineDescription": "Users should be able to reschedule appointments.",
                "uploadSessionId": session_id,
                "topK": 3,
            },
        )
        assert generate_response.status_code == 200
        generation_id = generate_response.json()["generationId"]

        debug_response = await client.get(f"/api/v1/generation/debug/{generation_id}")

        assert debug_response.status_code == 200
        debug_body = debug_response.json()
        assert debug_body["generationId"] == generation_id
        assert debug_body["feature"] == "Appointments"
        assert debug_body["redmineTicket"] == "12345"
        assert debug_body["retrieval"] == {
            "workflowChunks": 1,
            "historicalTestCases": 0,
            "historicalIssues": 0,
            "uploadedDocuments": 1,
        }
        # `requestedTestCases`/`countTargetMet` depend on whatever
        # MAX_GENERATED_TEST_CASES the environment running this test
        # suite has configured (see the dedicated, pinned-config tests
        # below) — this test's own purpose is just proving the debug
        # endpoint reads back exactly what `/generate` persisted, so it
        # only asserts the fields that don't vary with that setting.
        assert debug_body["metadata"]["topK"] == 3
        assert debug_body["metadata"]["uploadSessionId"] == session_id
        assert debug_body["metadata"]["generatedTestCases"] == 1
        assert "Workflow body text." in debug_body["prompt"]
        assert "Uploaded doc body text." in debug_body["prompt"]
        assert debug_body["response"] == _VALID_AI_RESPONSE
        assert len(debug_body["testCases"]) == 1
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_returns_422_for_invalid_ai_json(client, tmp_path):
    _override_generate_dependencies(tmp_path, chat_response_content="not valid json")

    try:
        response = await client.post(
            "/api/v1/generate",
            json={"feature": "Appointments", "redmineId": "12345", "redmineDescription": "Description text."},
        )

        assert response.status_code == 422
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_returns_422_for_schema_violation(client, tmp_path):
    invalid_response = json.dumps({"testCases": [{"requirementId": "REQ-1"}]})
    _override_generate_dependencies(tmp_path, chat_response_content=invalid_response)

    try:
        response = await client.post(
            "/api/v1/generate",
            json={"feature": "Appointments", "redmineId": "12345", "redmineDescription": "Description text."},
        )

        assert response.status_code == 422
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_returns_500_for_openai_failure(client, tmp_path):
    _override_generate_dependencies(tmp_path, chat_response_content=None)
    app.dependency_overrides[get_openai_client] = lambda: FakeOpenAIClient(
        chat_fail_with=RuntimeError("simulated API error")
    )

    try:
        response = await client.post(
            "/api/v1/generate",
            json={"feature": "Appointments", "redmineId": "12345", "redmineDescription": "Description text."},
        )

        assert response.status_code == 500
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_does_not_create_history_file_on_failure(client, tmp_path):
    _override_generate_dependencies(tmp_path, chat_response_content="not valid json")

    try:
        response = await client.post(
            "/api/v1/generate",
            json={"feature": "Appointments", "redmineId": "12345", "redmineDescription": "Description text."},
        )
        assert response.status_code == 422

        generation_root = tmp_path / "history" / "generation"
        assert not generation_root.exists() or list(generation_root.iterdir()) == []
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_works_without_upload_session_id(client, tmp_path):
    _override_generate_dependencies(tmp_path)

    try:
        response = await client.post(
            "/api/v1/generate",
            json={"feature": "Appointments", "redmineId": "12345", "redmineDescription": "Description text."},
        )

        assert response.status_code == 200
        generation_id = response.json()["generationId"]

        debug_response = await client.get(f"/api/v1/generation/debug/{generation_id}")
        assert debug_response.json()["metadata"]["uploadSessionId"] is None
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_never_calls_openai_chat_more_than_once(client, tmp_path):
    fake_client = _override_generate_dependencies(tmp_path)

    try:
        response = await client.post(
            "/api/v1/generate",
            json={"feature": "Appointments", "redmineId": "12345", "redmineDescription": "Description text."},
        )

        assert response.status_code == 200
        assert len(fake_client.chat_calls) == 1
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_accepts_a_missing_redmine_ticket(client, tmp_path):
    _override_generate_dependencies(tmp_path)

    try:
        response = await client.post("/api/v1/generate", json={"feature": "Appointments"})

        assert response.status_code == 200
        generation_id = response.json()["generationId"]

        debug_response = await client.get(f"/api/v1/generation/debug/{generation_id}")
        assert debug_response.json()["redmineTicket"] == ""
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_reported_regression_feature_only_no_ticket_no_description_no_upload(
    client, tmp_path
):
    """Exact reported failing scenario: Feature selected, Redmine Ticket
    ID empty, Optional Description empty, no documents uploaded — the
    request body this produces (verified in
    frontend/src/api/client.test.js) omits `redmineId`, `redmineDescription`,
    `optionalDescription`, and `uploadSessionId` entirely, exactly like this.
    """
    _override_generate_dependencies(tmp_path)

    try:
        response = await client.post(
            "/api/v1/generate",
            json={"feature": "Contact and Sticket Log", "requestId": "gen-request-regression"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        generation_id = body["generationId"]
        assert body["generatedTestCases"] == 1

        debug_response = await client.get(f"/api/v1/generation/debug/{generation_id}")
        debug_body = debug_response.json()
        assert debug_body["feature"] == "Contact and Sticket Log"
        assert debug_body["redmineTicket"] == ""
        assert debug_body["metadata"]["uploadSessionId"] is None
        assert "No Redmine ticket description was provided" in debug_body["prompt"]
    finally:
        _clear_generate_dependency_overrides()


def _test_cases_response(count: int, coverage_note: str | None = None) -> str:
    return json.dumps(
        {
            "testCases": [
                {
                    "requirementId": f"REQ-{i}",
                    "testCaseId": f"TC-{i}",
                    "testCaseTitle": f"Test case {i}",
                    "priority": "High",
                    "testSuite": "Appointments",
                    "preconditions": "User is logged in.",
                    "steps": [{"stepNo": 1, "action": "Do something.", "expectedResult": "Something happens."}],
                    "postConditions": "State updated.",
                    "automationStatus": "Not Automated",
                    "testType": "Functional",
                    "tags": ["Appointments"],
                }
                for i in range(1, count + 1)
            ],
            "coverageNote": coverage_note,
        }
    )


async def test_generate_endpoint_truncates_test_cases_exceeding_the_configured_maximum(client, tmp_path):
    """`len(testCases) <= MAX_GENERATED_TEST_CASES` is enforced
    regardless of what the model actually returns — pinned to a
    specific configured value (5), never the ambient `.env`.
    """
    _override_generate_dependencies_with_max_test_cases(
        tmp_path, max_test_cases=5, chat_response_content=_test_cases_response(8)
    )

    try:
        response = await client.post(
            "/api/v1/generate",
            json={"feature": "Appointments", "redmineId": "12345", "redmineDescription": "Description text."},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["generatedTestCases"] == 5
        assert body["requestedTestCases"] == 5
        assert body["countTargetMet"] is True
        assert len(body["testCases"]) == 5
        assert [tc["testCaseId"] for tc in body["testCases"]] == ["TC-1", "TC-2", "TC-3", "TC-4", "TC-5"]
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_returns_exactly_five_when_five_are_configured_and_supported(client, tmp_path):
    _override_generate_dependencies_with_max_test_cases(
        tmp_path, max_test_cases=5, chat_response_content=_test_cases_response(5)
    )

    try:
        response = await client.post(
            "/api/v1/generate",
            json={"feature": "Appointments", "redmineId": "12345", "redmineDescription": "Description text."},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["requestedTestCases"] == 5
        assert body["generatedTestCases"] == 5
        assert body["countTargetMet"] is True
        assert body["coverageNote"] is None
        assert len(body["testCases"]) == 5
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_returns_exactly_ten_when_ten_are_configured_and_supported(client, tmp_path):
    _override_generate_dependencies_with_max_test_cases(
        tmp_path, max_test_cases=10, chat_response_content=_test_cases_response(10)
    )

    try:
        response = await client.post(
            "/api/v1/generate",
            json={"feature": "Appointments", "redmineId": "12345", "redmineDescription": "Description text."},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["requestedTestCases"] == 10
        assert body["generatedTestCases"] == 10
        assert body["countTargetMet"] is True
        assert len(body["testCases"]) == 10
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_reports_target_not_met_when_the_model_returns_fewer_than_requested(client, tmp_path):
    """The model returning fewer than requested is treated as a valid,
    expected outcome (not padded, not an error) — `countTargetMet` and
    `coverageNote` are how a caller tells this apart from a real
    failure, since `generatedTestCases` alone can't.
    """
    _override_generate_dependencies_with_max_test_cases(
        tmp_path,
        max_test_cases=10,
        chat_response_content=_test_cases_response(
            4, coverage_note="Only 4 genuinely distinct scenarios are supported by the retrieved workflow context."
        ),
    )

    try:
        response = await client.post(
            "/api/v1/generate",
            json={"feature": "Appointments", "redmineId": "12345", "redmineDescription": "Description text."},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["requestedTestCases"] == 10
        assert body["generatedTestCases"] == 4
        assert body["countTargetMet"] is False
        assert (
            body["coverageNote"]
            == "Only 4 genuinely distinct scenarios are supported by the retrieved workflow context."
        )
        assert len(body["testCases"]) == 4
        # No padding: exactly the 4 the model returned, never fabricated
        # duplicates reaching toward 10.
        assert [tc["testCaseId"] for tc in body["testCases"]] == ["TC-1", "TC-2", "TC-3", "TC-4"]
        assert len({tc["testCaseId"] for tc in body["testCases"]}) == 4
    finally:
        _clear_generate_dependency_overrides()


async def test_generate_endpoint_never_fabricates_or_duplicates_test_cases_to_reach_the_configured_maximum(
    client, tmp_path
):
    _override_generate_dependencies_with_max_test_cases(
        tmp_path, max_test_cases=10, chat_response_content=_test_cases_response(3)
    )

    try:
        response = await client.post(
            "/api/v1/generate",
            json={"feature": "Appointments", "redmineId": "12345", "redmineDescription": "Description text."},
        )

        assert response.status_code == 200
        body = response.json()
        # Never padded up toward the configured maximum.
        assert len(body["testCases"]) == 3
        assert body["generatedTestCases"] == 3
        titles = [tc["testCaseTitle"] for tc in body["testCases"]]
        assert len(titles) == len(set(titles))
    finally:
        _clear_generate_dependency_overrides()


async def test_generation_progress_endpoint_returns_null_stage_for_an_unknown_request_id(client, tmp_path):
    _override_generate_dependencies(tmp_path)

    try:
        response = await client.get("/api/v1/generate/progress/never-started")

        assert response.status_code == 200
        assert response.json() == {"stage": None}
    finally:
        _clear_generate_dependency_overrides()


async def test_generation_progress_endpoint_reflects_real_progress_while_generation_is_in_flight(client, tmp_path):
    """`POST /generate` is run in a worker thread specifically so this
    concurrent poll can still be served while it's in flight — proof the
    progress endpoint reports something real, not a value estimated from
    elapsed time.
    """
    fake_client = _override_generate_dependencies(
        tmp_path, chat_response_content=_VALID_AI_RESPONSE, chat_delay_seconds=0.4
    )

    try:
        request_id = "req-live-progress"
        generate_task = asyncio.create_task(
            client.post(
                "/api/v1/generate",
                json={
                    "feature": "Appointments",
                    "redmineId": "12345",
                    "redmineDescription": "Description text.",
                    "requestId": request_id,
                },
            )
        )

        observed_stages = set()
        for _ in range(20):
            await asyncio.sleep(0.05)
            progress_response = await client.get(f"/api/v1/generate/progress/{request_id}")
            stage = progress_response.json()["stage"]
            if stage:
                observed_stages.add(stage)
            if generate_task.done():
                break

        response = await generate_task
        assert response.status_code == 200
        assert observed_stages, "expected to observe at least one real in-flight stage"
        assert observed_stages <= {
            "retrievingDocuments",
            "buildingPrompt",
            "generatingTestCases",
            "validatingResponse",
            "savingHistory",
        }

        # Cleared once the request finishes.
        final_progress = await client.get(f"/api/v1/generate/progress/{request_id}")
        assert final_progress.json() == {"stage": None}
        assert fake_client.chat_calls
    finally:
        _clear_generate_dependency_overrides()
