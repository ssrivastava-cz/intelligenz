"""Integration tests for `POST /knowledge-assistant/ask`,
`POST /knowledge-assistant/feedback`, and
`GET /knowledge-assistant/debug/{generation_id}`. Source of Truth
chunks are produced through the real `/index-feature/{feature}`
endpoint (exactly like `tests/integration/test_index_service.py`); only
OpenAI is faked (`tests/fakes.py`) — the same fake client stands in for
both the query-embedding call and the chat completion call. Persistence
goes through the real `FileSystemGenerationRepository`, rooted at
`tmp_path` — never the real `backend/database/`.
"""
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.dependencies import get_chroma_client, get_generation_repository, get_history_service, get_openai_client
from app.main import app
from app.repositories.filesystem_generation_repository import FileSystemGenerationRepository
from app.services.history_service import HistoryService
from tests.fakes import FakeOpenAIClient

_VALID_ANSWER_NO_SOURCES = json.dumps({"answer": "The documentation does not cover this.", "sourceDocuments": []})


def _valid_answer(
    source_documents: list[str] | None = None,
    answer: str = "According to the workflow, bridged contacts are merged automatically.",
) -> str:
    return json.dumps({"answer": answer, "sourceDocuments": source_documents or []})


def _write_workflow_document(tmp_path, feature: str, filename: str, content: str) -> None:
    root = tmp_path / "source_of_truth" / feature / "workflows"
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text(content, encoding="utf-8")


def _override_knowledge_assistant_dependencies(tmp_path, fake_openai_client: FakeOpenAIClient | None = None):
    # Same default dimension `FakeOpenAIClient()` already uses (3) — the
    # same one every other integration test in this suite that goes
    # through the real DI graph relies on, since `get_vector_store_service`
    # always uses the *same* fixed collection name
    # (`settings.chroma_collection_name`) regardless of which test built
    # it, and `chromadb.EphemeralClient()` instances share underlying
    # storage process-wide — a different dimension here would collide
    # with whatever another integration test file already wrote there.
    client = fake_openai_client or FakeOpenAIClient(chat_response_content=_VALID_ANSWER_NO_SOURCES)
    app.dependency_overrides[get_openai_client] = lambda: client
    app.dependency_overrides[get_chroma_client] = lambda: chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    app.dependency_overrides[get_history_service] = lambda: HistoryService(history_root=tmp_path / "history")
    app.dependency_overrides[get_generation_repository] = lambda: FileSystemGenerationRepository(
        database_root=tmp_path / "database"
    )
    return client


def _clear_knowledge_assistant_dependency_overrides() -> None:
    app.dependency_overrides.pop(get_openai_client, None)
    app.dependency_overrides.pop(get_chroma_client, None)
    app.dependency_overrides.pop(get_history_service, None)
    app.dependency_overrides.pop(get_generation_repository, None)


async def _ask(client, question: str = "How does Contact Log work?", feature: str = "Contact Log"):
    """Posts a question (no `generationId` in the payload — the backend
    is the sole authority for it now) and returns the parsed response
    body, so callers can read the backend-assigned `generationId` off
    it rather than assuming one.
    """
    response = await client.post(
        "/api/v1/knowledge-assistant/ask", json={"userQuestion": question, "feature": feature}
    )
    return response


# --- POST /ask: the new answer/source_documents/usage response shape ---


async def test_ask_endpoint_returns_the_answer_and_source_documents(client, tmp_path):
    _write_workflow_document(
        tmp_path,
        "Contact Log",
        "Contact_Log_Workflow.md",
        "# Bridged Contacts\n\nBridged contacts are merged automatically when matching criteria align.",
    )
    fake_client = FakeOpenAIClient(chat_response_content=_valid_answer(["Contact_Log_Workflow.md"]))
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        await client.post("/api/v1/index-feature/Contact Log")

        response = await _ask(client, "How does Contact Log handle bridged contacts?")

        assert response.status_code == 200
        body = response.json()
        assert body["generationId"]
        assert body["question"] == "How does Contact Log handle bridged contacts?"
        assert body["answer"] == "According to the workflow, bridged contacts are merged automatically."
        assert body["sourceDocuments"] == ["Contact_Log_Workflow.md"]
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_ask_endpoint_never_generates_the_generation_id_from_the_request_body(client, tmp_path):
    """Section 2/4: the frontend no longer supplies a `generationId` at
    all — this proves the endpoint doesn't require or even accept one
    from the request; it's purely a response field.
    """
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        response = await client.post(
            "/api/v1/knowledge-assistant/ask",
            json={"userQuestion": "How does Contact Log work?", "feature": "Contact Log"},
        )

        assert response.status_code == 200
        assert response.json()["generationId"]
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_ask_endpoint_returns_usage_with_actual_and_estimated_tokens(client, tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_VALID_ANSWER_NO_SOURCES, chat_prompt_tokens=3200, chat_completion_tokens=580
    )
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        response = await _ask(client)

        usage = response.json()["usage"]
        assert usage["inputTokens"] == 3200
        assert usage["outputTokens"] == 580
        assert usage["totalTokens"] == 3780
        assert usage["estimatedInputTokens"] > 0
        assert usage["inputCostInr"] >= 0
        assert usage["outputCostInr"] >= 0
        assert usage["totalCostInr"] >= 0
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_ask_endpoint_never_returns_the_full_prompt_or_raw_chunk_contents(client, tmp_path):
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        response = await _ask(client)

        body = response.json()
        assert set(body.keys()) == {"generationId", "question", "answer", "sourceDocuments", "usage"}
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_ask_endpoint_uses_the_configured_chat_model(client, tmp_path):
    """Section 2: reuse the project's single configured chat model — no
    second Knowledge-Assistant-specific model configuration exists."""
    from app.config.settings import get_settings

    fake_client = FakeOpenAIClient(chat_response_content=_VALID_ANSWER_NO_SOURCES)
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        await _ask(client)

        [call] = fake_client.chat_calls
        assert call["model"] == get_settings().llm_model
    finally:
        _clear_knowledge_assistant_dependency_overrides()


# --- Failure behaviour, at the HTTP layer ---


async def test_ask_endpoint_returns_an_error_status_when_openai_fails(client, tmp_path):
    fake_client = FakeOpenAIClient(chat_fail_with=RuntimeError("connection reset"))
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        response = await _ask(client)

        assert response.status_code >= 500
        assert len(fake_client.chat_calls) == 1
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_ask_endpoint_returns_a_client_error_when_the_response_is_invalid(client, tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content="not valid json")
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        response = await _ask(client)

        assert response.status_code == 422
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_ask_endpoint_failure_does_not_persist_a_successful_response(client, tmp_path):
    fake_client = FakeOpenAIClient(chat_fail_with=RuntimeError("connection reset"))
    generation_repository = _override_knowledge_assistant_dependencies_and_get_repository(tmp_path, fake_client)

    try:
        await _ask(client)

        [item] = generation_repository.list_generations()
        debug_response = await client.get(f"/api/v1/knowledge-assistant/debug/{item.generation_id}")

        assert debug_response.status_code == 200
        assert debug_response.json()["response"] is None
    finally:
        _clear_knowledge_assistant_dependency_overrides()


def _override_knowledge_assistant_dependencies_and_get_repository(tmp_path, fake_openai_client=None):
    """Like `_override_knowledge_assistant_dependencies`, but also
    returns the exact `FileSystemGenerationRepository` instance
    installed — needed by tests that must recover a `generationId`
    that was never returned in an HTTP response (e.g. after a failed
    `POST /ask`, whose error response carries no `generationId` field).
    """
    _override_knowledge_assistant_dependencies(tmp_path, fake_openai_client)
    return FileSystemGenerationRepository(database_root=tmp_path / "database")


# --- POST /feedback ---


async def test_feedback_endpoint_records_a_good_evaluation(client, tmp_path):
    generation_repository = _override_knowledge_assistant_dependencies_and_get_repository(tmp_path)

    try:
        ask_response = await _ask(client)
        generation_id = ask_response.json()["generationId"]

        response = await client.post(
            "/api/v1/knowledge-assistant/feedback",
            json={"generationId": generation_id, "evaluation": "GOOD"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["generationId"] == generation_id
        assert body["evaluation"] == "GOOD"
        assert generation_repository.get_generation(generation_id).feedback.evaluation == "GOOD"
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_feedback_endpoint_records_a_bad_evaluation_with_reason_and_description(client, tmp_path):
    generation_repository = _override_knowledge_assistant_dependencies_and_get_repository(tmp_path)

    try:
        ask_response = await _ask(client)
        generation_id = ask_response.json()["generationId"]

        response = await client.post(
            "/api/v1/knowledge-assistant/feedback",
            json={
                "generationId": generation_id,
                "evaluation": "BAD",
                "reason": "MISSING_INFORMATION",
                "description": "Missed the Bridge functionality.",
            },
        )

        assert response.status_code == 200
        feedback = generation_repository.get_generation(generation_id).feedback
        assert feedback.evaluation == "BAD"
        assert feedback.reason == "MISSING_INFORMATION"
        assert feedback.description == "Missed the Bridge functionality."
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_feedback_endpoint_returns_404_for_an_unknown_generation_id(client, tmp_path):
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        response = await client.post(
            "/api/v1/knowledge-assistant/feedback",
            json={"generationId": "gen_does_not_exist", "evaluation": "GOOD"},
        )

        assert response.status_code == 404
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_feedback_endpoint_makes_no_ai_calls(client, tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_ANSWER_NO_SOURCES)
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        ask_response = await _ask(client)
        generation_id = ask_response.json()["generationId"]
        calls_after_ask = len(fake_client.calls)
        chat_calls_after_ask = len(fake_client.chat_calls)

        await client.post(
            "/api/v1/knowledge-assistant/feedback", json={"generationId": generation_id, "evaluation": "GOOD"}
        )

        assert len(fake_client.calls) == calls_after_ask
        assert len(fake_client.chat_calls) == chat_calls_after_ask
    finally:
        _clear_knowledge_assistant_dependency_overrides()


# --- GET /recent: backs "Recent Questions and Answers", real generations only ---


async def test_recent_endpoint_returns_an_empty_list_when_no_generations_exist(client, tmp_path):
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        response = await client.get("/api/v1/knowledge-assistant/recent")

        assert response.status_code == 200
        assert response.json() == []
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_recent_endpoint_reconstructs_question_answer_and_source_documents(client, tmp_path):
    _write_workflow_document(
        tmp_path,
        "Contact Log",
        "Contact_Log_Workflow.md",
        "# Bridged Contacts\n\nBridged contacts are merged automatically when matching criteria align.",
    )
    fake_client = FakeOpenAIClient(chat_response_content=_valid_answer(["Contact_Log_Workflow.md"]))
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        await client.post("/api/v1/index-feature/Contact Log")
        ask_response = await _ask(client, "How does Contact Log handle bridged contacts?")
        generation_id = ask_response.json()["generationId"]

        response = await client.get("/api/v1/knowledge-assistant/recent")

        assert response.status_code == 200
        [item] = response.json()
        assert item["generationId"] == generation_id
        assert item["question"] == "How does Contact Log handle bridged contacts?"
        assert item["answer"] == "According to the workflow, bridged contacts are merged automatically."
        assert item["sourceDocuments"] == ["Contact_Log_Workflow.md"]
        assert item["createdAt"]
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_recent_endpoint_returns_generations_newest_first(client, tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_ANSWER_NO_SOURCES)
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        await _ask(client, "First question asked.")
        await _ask(client, "Second question asked.")

        response = await client.get("/api/v1/knowledge-assistant/recent")

        questions = [item["question"] for item in response.json()]
        assert questions == ["Second question asked.", "First question asked."]
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_recent_endpoint_excludes_generations_without_a_successful_response(client, tmp_path):
    fake_client = FakeOpenAIClient(chat_fail_with=RuntimeError("connection reset"))
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        await _ask(client, "This question never gets an answer.")

        response = await client.get("/api/v1/knowledge-assistant/recent")

        assert response.status_code == 200
        assert response.json() == []
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_recent_endpoint_respects_the_limit_query_parameter(client, tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_ANSWER_NO_SOURCES)
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        await _ask(client, "Question one.")
        await _ask(client, "Question two.")
        await _ask(client, "Question three.")

        response = await client.get("/api/v1/knowledge-assistant/recent", params={"limit": 2})

        assert len(response.json()) == 2
    finally:
        _clear_knowledge_assistant_dependency_overrides()


# --- Debug endpoint: reads persisted data, never re-runs anything ---


async def test_debug_endpoint_returns_the_retrieved_documents(client, tmp_path):
    _write_workflow_document(
        tmp_path,
        "Contact Log",
        "Contact_Log_Workflow.md",
        "# Bridged Contacts\n\nBridged contacts are merged automatically when matching criteria align.",
    )
    fake_client = FakeOpenAIClient(chat_response_content=_valid_answer(["Contact_Log_Workflow.md"]))
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        index_response = await client.post("/api/v1/index-feature/Contact Log")
        assert index_response.status_code == 200

        ask_response = await _ask(client, "How does Contact Log handle bridged contacts?")
        assert ask_response.status_code == 200
        generation_id = ask_response.json()["generationId"]

        debug_response = await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")

        assert debug_response.status_code == 200
        body = debug_response.json()
        assert body["generationId"] == generation_id
        assert body["question"] == "How does Contact Log handle bridged contacts?"
        assert "Contact_Log_Workflow.md" in body["retrieval"]["workflow"]["documents"]
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_returns_similarity_and_chunk_detail_without_the_embedding_vector(client, tmp_path):
    _write_workflow_document(
        tmp_path,
        "Contact Log",
        "Contact_Log_Workflow.md",
        "# Bridged Contacts\n\nBridged contacts are merged automatically when matching criteria align.",
    )
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        await client.post("/api/v1/index-feature/Contact Log")
        ask_response = await _ask(client)
        generation_id = ask_response.json()["generationId"]

        body = (await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")).json()

        [chunk] = body["retrieval"]["workflow"]["chunks"]
        assert chunk["chunkId"]
        assert chunk["documentId"]
        assert chunk["documentName"] == "Contact_Log_Workflow.md"
        assert chunk["artifactType"] == "WORKFLOW"
        assert chunk["sectionHeading"] == "Bridged Contacts"
        assert isinstance(chunk["similarityScore"], float)
        assert "chunkNumber" in chunk
        assert "merged" in chunk
        # No embedding vector or raw distance ever exposed.
        assert "embedding" not in chunk
        assert "vector" not in chunk
        assert "distance" not in chunk
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_returns_pipeline_stage_counts_for_every_source(client, tmp_path):
    _write_workflow_document(
        tmp_path,
        "Contact Log",
        "Contact_Log_Workflow.md",
        "# Bridged Contacts\n\nBridged contacts are merged automatically when matching criteria align.",
    )
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        await client.post("/api/v1/index-feature/Contact Log")
        ask_response = await _ask(client)
        generation_id = ask_response.json()["generationId"]

        retrieval = (await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")).json()["retrieval"]

        assert retrieval["workflow"]["candidateRetrieved"] == 1
        assert retrieval["workflow"]["finalChunks"] == 1
        assert retrieval["workflow"]["mergedChunks"] == 1
        # Nothing indexed for the other sources.
        assert retrieval["historicalTestCases"]["candidateRetrieved"] == 0
        assert retrieval["historicalTestCases"]["chunks"] == []
        assert retrieval["historicalIssues"]["candidateRetrieved"] == 0
        assert retrieval["uploadedDocuments"]["candidateRetrieved"] == 0
    finally:
        _clear_knowledge_assistant_dependency_overrides()


# --- Debug endpoint: Hybrid Retrieval diagnostics ---


async def test_debug_endpoint_returns_hybrid_retrieval_diagnostics(client, tmp_path):
    _write_workflow_document(
        tmp_path,
        "Contact Log",
        "Contact_Log_Workflow.md",
        "# Bridged Contacts\n\nBridged contacts are merged automatically when matching criteria align.",
    )
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        await client.post("/api/v1/index-feature/Contact Log")
        ask_response = await _ask(client, "How does Contact Log handle bridged contacts?")
        generation_id = ask_response.json()["generationId"]

        diagnostics = (await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")).json()[
            "retrievalDiagnostics"
        ]

        assert diagnostics is not None
        assert diagnostics["finalChunkCount"] == 1
        assert diagnostics["vectorCandidateCount"] >= 1
        assert diagnostics["duplicatesRemoved"] == 0
        assert diagnostics["estimatedPromptTokens"] > 0
        assert diagnostics["totalRetrievalMs"] >= 0

        [candidate] = diagnostics["candidates"]
        assert candidate["chunkId"]
        assert candidate["documentName"] == "Contact_Log_Workflow.md"
        assert candidate["artifactType"] == "WORKFLOW"
        assert candidate["rrfScore"] > 0
        assert candidate["rerankerScore"] > 0
        assert candidate["finalRank"] == 1
        assert candidate["selected"] is True

        [final_chunk] = diagnostics["finalContext"]
        assert final_chunk["documentName"] == "Contact_Log_Workflow.md"
        assert "Bridged contacts" in final_chunk["text"]
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_diagnostics_never_expose_an_embedding_vector(client, tmp_path):
    _write_workflow_document(
        tmp_path,
        "Contact Log",
        "Contact_Log_Workflow.md",
        "# Bridged Contacts\n\nBridged contacts are merged automatically when matching criteria align.",
    )
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        await client.post("/api/v1/index-feature/Contact Log")
        ask_response = await _ask(client, "How does Contact Log handle bridged contacts?")
        generation_id = ask_response.json()["generationId"]

        response = await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")

        body_text = response.text.lower()
        assert '"embedding"' not in body_text
        assert '"vector"' not in body_text
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_does_not_perform_retrieval_again(client, tmp_path):
    """Read-only: the debug endpoint must show exactly what `ask()`
    already persisted, never re-run retrieval — proven here by an
    embedding-call count that doesn't move across repeated debug reads,
    exactly like the existing OpenAI call-count guarantee."""
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_ANSWER_NO_SOURCES)
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        ask_response = await _ask(client)
        generation_id = ask_response.json()["generationId"]
        embedding_calls_after_ask = len(fake_client.calls)

        first = await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")
        second = await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")

        assert len(fake_client.calls) == embedding_calls_after_ask
        assert first.json()["retrievalDiagnostics"] == second.json()["retrievalDiagnostics"]
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_retrieval_diagnostics_is_null_when_retrieval_never_ran(client, tmp_path):
    """Section 10: a generation that failed before retrieval even
    started (input validation) has no retrieval diagnostics to show —
    `None`, not a fabricated empty structure."""
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_ANSWER_NO_SOURCES)
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        response = await client.post(
            "/api/v1/knowledge-assistant/ask", json={"userQuestion": "   ", "feature": "Contact Log"}
        )
        assert response.status_code >= 400
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_returns_prompt_token_and_estimated_cost_information(client, tmp_path):
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        ask_response = await _ask(client)
        generation_id = ask_response.json()["generationId"]

        prompt = (await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")).json()["prompt"]

        assert prompt["promptVersion"] == "knowledge-assistant-v1"
        assert prompt["inputTokens"] > 0
        assert prompt["estimatedInputTokens"] > 0
        assert prompt["estimatedInputCostInr"] >= 0
        # A debug endpoint, not a general read-back — never the raw prompt text.
        assert set(prompt.keys()) == {"promptVersion", "inputTokens", "estimatedInputTokens", "estimatedInputCostInr"}
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_exposes_the_actual_response_and_actual_usage(client, tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_valid_answer(answer="Here is the grounded answer."),
        chat_prompt_tokens=3200,
        chat_completion_tokens=580,
    )
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        ask_response = await _ask(client)
        generation_id = ask_response.json()["generationId"]

        body = (await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")).json()

        assert body["response"]["answer"] == "Here is the grounded answer."
        assert body["response"]["usage"]["inputTokens"] == 3200
        assert body["response"]["usage"]["outputTokens"] == 580
        assert body["response"]["usage"]["totalTokens"] == 3780
        assert body["response"]["usage"]["totalCostInr"] >= 0
        assert body["error"] is None
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_exposes_the_detailed_token_breakdown(client, tmp_path):
    """The reported bug: gen_20260813_9c771 showed output_tokens=1094
    for a short visible answer — the debug endpoint must expose
    reasoning/visible-answer/structured-output-overhead tokens so the
    real source of a surprising output_tokens figure is inspectable."""
    fake_client = FakeOpenAIClient(
        chat_response_content=_valid_answer(answer="Short visible answer."),
        chat_completion_tokens=1094,
        chat_reasoning_tokens=960,
        chat_cached_tokens=128,
    )
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        ask_response = await _ask(client)
        generation_id = ask_response.json()["generationId"]

        usage = (await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")).json()["response"]["usage"]

        assert usage["outputTokens"] == 1094
        assert usage["reasoningTokens"] == 960
        assert usage["cachedTokens"] == 128
        assert 0 < usage["visibleAnswerTokens"] < 20
        assert usage["structuredOutputOverheadTokens"] >= 0
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_ask_endpoint_response_does_not_expose_the_detailed_token_breakdown(client, tmp_path):
    """The detailed breakdown is debug-endpoint (and Usage Dashboard)
    only — the normal `/ask` response shape is unchanged."""
    fake_client = FakeOpenAIClient(
        chat_response_content=_VALID_ANSWER_NO_SOURCES, chat_completion_tokens=1094, chat_reasoning_tokens=960
    )
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        response = await _ask(client)

        assert set(response.json()["usage"].keys()) == {
            "inputTokens",
            "outputTokens",
            "totalTokens",
            "estimatedInputTokens",
            "inputCostInr",
            "outputCostInr",
            "totalCostInr",
        }
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_exposes_none_for_reasoning_and_cached_tokens_when_openai_omits_them(client, tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_ANSWER_NO_SOURCES)
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        ask_response = await _ask(client)
        generation_id = ask_response.json()["generationId"]

        usage = (await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")).json()["response"]["usage"]

        assert usage["reasoningTokens"] is None
        assert usage["cachedTokens"] is None
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_exposes_error_information_for_a_failed_generation(client, tmp_path):
    fake_client = FakeOpenAIClient(chat_fail_with=RuntimeError("connection reset"))
    generation_repository = _override_knowledge_assistant_dependencies_and_get_repository(tmp_path, fake_client)

    try:
        await _ask(client)
        [item] = generation_repository.list_generations()

        body = (await client.get(f"/api/v1/knowledge-assistant/debug/{item.generation_id}")).json()

        assert body["response"] is None
        assert body["error"] is not None
        assert body["error"]["stage"] == "OPENAI_REQUEST"
        assert body["error"]["errorType"] == "ExternalServiceError"
        assert "connection reset" in body["error"]["errorMessage"]
        assert body["error"]["createdAt"]
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_response_is_null_before_a_successful_answer_exists(client, tmp_path):
    fake_client = FakeOpenAIClient(chat_fail_with=RuntimeError("connection reset"))
    generation_repository = _override_knowledge_assistant_dependencies_and_get_repository(tmp_path, fake_client)

    try:
        await _ask(client)
        [item] = generation_repository.list_generations()

        body = (await client.get(f"/api/v1/knowledge-assistant/debug/{item.generation_id}")).json()

        assert body["response"] is None
        # Question/retrieval/prompt still show, since those steps did succeed.
        assert body["question"] == "How does Contact Log work?"
        assert body["prompt"]["estimatedInputTokens"] > 0
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_never_calls_openai(client, tmp_path):
    """The debug endpoint must not itself trigger any OpenAI call — only
    `POST /ask` (called once, beforehand) does."""
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_ANSWER_NO_SOURCES)
    _override_knowledge_assistant_dependencies(tmp_path, fake_client)

    try:
        ask_response = await _ask(client)
        generation_id = ask_response.json()["generationId"]
        calls_after_ask = len(fake_client.chat_calls)
        embedding_calls_after_ask = len(fake_client.calls)

        await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")
        await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")

        assert len(fake_client.chat_calls) == calls_after_ask
        assert len(fake_client.calls) == embedding_calls_after_ask
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_response_never_contains_the_word_embedding_as_a_data_field(client, tmp_path):
    _write_workflow_document(
        tmp_path,
        "Contact Log",
        "Contact_Log_Workflow.md",
        "# Bridged Contacts\n\nBridged contacts are merged automatically when matching criteria align.",
    )
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        await client.post("/api/v1/index-feature/Contact Log")
        ask_response = await _ask(client)
        generation_id = ask_response.json()["generationId"]

        response = await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")

        body_text = response.text.lower()
        assert '"embedding"' not in body_text
        assert '"vector"' not in body_text
        assert '"embeddingvector"' not in body_text
    finally:
        _clear_knowledge_assistant_dependency_overrides()


# --- missing context / not found handling ---


async def test_debug_endpoint_returns_404_for_an_unknown_generation_id(client, tmp_path):
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        response = await client.get("/api/v1/knowledge-assistant/debug/gen_does_not_exist")

        assert response.status_code == 404
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_debug_endpoint_handles_a_question_with_no_retrieved_context_gracefully(client, tmp_path):
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        ask_response = await _ask(client, "Is there documentation for this?", feature="Nonexistent Feature")
        generation_id = ask_response.json()["generationId"]

        response = await client.get(f"/api/v1/knowledge-assistant/debug/{generation_id}")

        assert response.status_code == 200
        retrieval = response.json()["retrieval"]
        assert retrieval["workflow"]["chunks"] == []
        assert retrieval["workflow"]["documents"] == []
    finally:
        _clear_knowledge_assistant_dependency_overrides()


async def test_ask_endpoint_writes_only_under_the_overridden_database_root(client, tmp_path):
    """`_protect_real_application_data` (conftest.py) guards
    `backend/data/`, `backend/source_of_truth/`, and `backend/uploads/`
    but not `backend/database/` — this checks that invariant directly
    for the Knowledge Assistant persistence path: everything `ask()`
    writes lands under the overridden `tmp_path`, never the real
    `backend/database/`. Doesn't assert the real directory is entirely
    absent (real, legitimate Knowledge Assistant usage against a
    developer's running server can populate it outside of any test
    run) — only that *this test's own* generation never appears there.
    """
    real_database_root = Path(__file__).resolve().parent.parent.parent / "database"
    _override_knowledge_assistant_dependencies(tmp_path)

    try:
        response = await _ask(client)
        generation_id = response.json()["generationId"]

        assert response.status_code == 200
        assert (tmp_path / "database" / "generations" / generation_id / "user_question.json").is_file()
        assert (tmp_path / "database" / "generations" / generation_id / "response.json").is_file()
        assert (tmp_path / "database" / "generations" / generation_id / "usage.json").is_file()
        assert not (real_database_root / "generations" / generation_id).exists()
    finally:
        _clear_knowledge_assistant_dependency_overrides()
