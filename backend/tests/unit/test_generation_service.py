"""Unit tests for GenerationService — the only component allowed to
call OpenAI Chat, orchestrating RetrievalService -> PromptBuilder ->
OpenAI Chat -> response validation -> CostCalculator ->
HistoryService.save_generation_history. Every dependency here is real
except OpenAI itself (the one boundary this test suite fakes outright,
per tests/fakes.py); ChromaDB is a real `chromadb.EphemeralClient`.
"""
import json
import uuid

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.core.exceptions import ExternalServiceError, ValidationError
from app.models.generated_test_case import GeneratedTestCasesResponse
from app.retrievers.text_tokenizer import TOKENIZER_VERSION, tokenize
from app.services.bm25_index_manager import BM25IndexManager
from app.services.cost_calculator import CostCalculator
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.history_service import HistoryService
from app.services.prompt_builder import PromptBuilder
from app.services.retrieval_service import RetrievalService
from app.services.vector_store_service import VectorStoreService
from tests.fakes import FakeOpenAIClient

_EMBEDDING_MODEL = "text-embedding-3-small"
_CHAT_MODEL = "gpt-5"
_INPUT_PRICE_PER_MILLION = 1.25
_OUTPUT_PRICE_PER_MILLION = 10.00
_USD_TO_INR_EXCHANGE_RATE = 87.0

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


def _make_service(tmp_path, fake_client: FakeOpenAIClient | None = None, max_generated_test_cases: int | None = None):
    client = fake_client if fake_client is not None else FakeOpenAIClient(chat_response_content=_VALID_AI_RESPONSE)
    embedding_service = EmbeddingService(client=client, model=_EMBEDDING_MODEL)

    chroma_client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)

    _bm25_manager = BM25IndexManager(
        index_dir=tmp_path / "bm25",
        collection_name=collection_name,
        tokenizer=tokenize,
        tokenizer_version=TOKENIZER_VERSION,
        chunking_signature="chunk_size=500,chunk_overlap=50",
    )
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        chroma_client=chroma_client,
        bm25_index_manager=_bm25_manager,
        upload_collection_prefix="uploaded_documents",
        default_top_k=5,
    )
    # Mirrors production DI wiring (`get_prompt_builder`/`get_generation_service`,
    # both reading the same `settings.max_generated_test_cases`): the
    # prompt's stated target and the service's enforced ceiling must
    # always be the same configured number, never two independently
    # drifting values.
    prompt_builder = (
        PromptBuilder(max_test_cases=max_generated_test_cases)
        if max_generated_test_cases is not None
        else PromptBuilder()
    )
    cost_calculator = CostCalculator(
        model=_CHAT_MODEL,
        input_price_per_million_tokens=_INPUT_PRICE_PER_MILLION,
        output_price_per_million_tokens=_OUTPUT_PRICE_PER_MILLION,
        usd_to_inr_exchange_rate=_USD_TO_INR_EXCHANGE_RATE,
    )
    history_service = HistoryService(history_root=tmp_path / "history")

    service_kwargs = {}
    if max_generated_test_cases is not None:
        service_kwargs["max_generated_test_cases"] = max_generated_test_cases

    service = GenerationService(
        retrieval_service=retrieval_service,
        prompt_builder=prompt_builder,
        embedding_service=embedding_service,
        openai_client=client,
        cost_calculator=cost_calculator,
        history_service=history_service,
        chat_model=_CHAT_MODEL,
        **service_kwargs,
    )
    return service, vector_store, client, history_service


def _generation_history_files(tmp_path) -> list:
    generation_root = tmp_path / "history" / "generation"
    return list(generation_root.iterdir()) if generation_root.is_dir() else []


def test_generate_returns_a_history_entry_with_parsed_test_cases(tmp_path):
    service, _, _, _ = _make_service(tmp_path)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert len(entry.test_cases) == 1
    assert entry.test_cases[0].test_case_id == "TC-1"
    assert entry.test_cases[0].steps[0].action == "Open appointment."
    assert entry.metadata.generated_test_cases == 1


def test_generate_saves_exactly_one_history_file_on_success(tmp_path):
    service, _, _, _ = _make_service(tmp_path)

    service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert len(_generation_history_files(tmp_path)) == 1


def test_generate_calls_openai_chat_exactly_once(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_AI_RESPONSE)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert len(fake_client.chat_calls) == 1


def test_generate_sends_the_built_prompt_as_the_chat_message_content(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_AI_RESPONSE)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [call] = fake_client.chat_calls
    assert call["messages"] == [{"role": "user", "content": entry.prompt}]
    assert call["model"] == _CHAT_MODEL
    # Structured Outputs: the model class itself is passed as
    # response_format, constraining the API to this exact schema rather
    # than a generic {"type": "json_object"} instruction.
    assert call["response_format"] is GeneratedTestCasesResponse


def test_generate_uses_actual_openai_usage_not_the_local_estimate(tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_VALID_AI_RESPONSE, chat_prompt_tokens=99_999, chat_completion_tokens=1_234
    )
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert entry.actual_usage.prompt_tokens == 99_999
    assert entry.actual_usage.completion_tokens == 1_234
    assert entry.actual_usage.total_tokens == 101_233
    # The local pre-generation estimate is a completely different (much
    # smaller) number for this short prompt — proof actual usage isn't
    # just a copy of the estimate.
    assert entry.estimated_usage.input_tokens != entry.actual_usage.prompt_tokens


def test_generate_computes_actual_cost_via_cost_calculator(tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_VALID_AI_RESPONSE, chat_prompt_tokens=100, chat_completion_tokens=50
    )
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert entry.actual_usage.input_cost.usd == pytest.approx(0.000125)
    assert entry.actual_usage.output_cost.usd == pytest.approx(0.0005)
    assert entry.actual_usage.total_cost.usd == pytest.approx(0.000625)
    assert entry.actual_usage.input_cost.inr == 0.01
    assert entry.actual_usage.output_cost.inr == 0.04
    assert entry.actual_usage.total_cost.inr == 0.05


def test_generate_persists_a_pricing_snapshot_matching_the_cost_calculator(tmp_path):
    service, _, _, _ = _make_service(tmp_path)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert entry.pricing.model == _CHAT_MODEL
    assert entry.pricing.input_price_per_million_tokens == _INPUT_PRICE_PER_MILLION
    assert entry.pricing.output_price_per_million_tokens == _OUTPUT_PRICE_PER_MILLION
    assert entry.pricing.usd_to_inr_exchange_rate == _USD_TO_INR_EXCHANGE_RATE


def test_generate_records_retrieval_summary_counts(tmp_path):
    service, vector_store, _, _ = _make_service(tmp_path)
    vector_store.replace_feature_chunks(
        feature="Appointments",
        ids=["a"],
        embeddings=[[1.0, 0.0, 0.0]],  # 3-dimensional to match FakeOpenAIClient's default query embedding
        documents=["Workflow text."],
        metadatas=[
            {
                "chunkId": "a",
                "artifactType": "WORKFLOW",
                "feature": "Appointments",
                "documentSource": "source_of_truth",
                "sourceFilename": "onboarding.md",
                "parserName": "MarkdownParser",
                "parserVersion": "1.0",
            }
        ],
    )

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert entry.retrieval.workflow_chunks == 1
    assert entry.retrieval.historical_test_cases == 0
    assert entry.retrieval.historical_issues == 0
    assert entry.retrieval.uploaded_documents == 0


def test_generate_records_metadata_top_k_and_upload_session_id(tmp_path):
    service, _, _, _ = _make_service(tmp_path)

    entry = service.generate(
        feature="Appointments",
        redmine_id="12345",
        redmine_description="Description text.",
        upload_session_id="sess-abc123",
        top_k=3,
    )

    assert entry.metadata.top_k == 3
    assert entry.metadata.upload_session_id == "sess-abc123"


def test_generate_uses_optional_description_as_the_retrieval_query_when_given(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_AI_RESPONSE)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    service.generate(
        feature="Appointments",
        redmine_id="12345",
        redmine_description="Redmine description text.",
        optional_description="User supplied description text.",
    )

    assert fake_client.calls == [["User supplied description text."]]


def test_generate_raises_validation_error_for_invalid_json_and_saves_no_history(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content="not valid json")
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError):
        service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert _generation_history_files(tmp_path) == []


def test_generate_raises_validation_error_for_schema_violation_and_saves_no_history(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=json.dumps({"testCases": [{"requirementId": "REQ-1"}]}))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError):
        service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert _generation_history_files(tmp_path) == []


def test_generate_raises_validation_error_for_missing_test_cases_key_and_saves_no_history(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=json.dumps({"somethingElse": []}))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError):
        service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert _generation_history_files(tmp_path) == []


def test_generate_raises_validation_error_for_empty_response_content_and_saves_no_history(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=None)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError):
        service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert _generation_history_files(tmp_path) == []


def test_generate_raises_validation_error_with_refusal_reason_and_saves_no_history(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=None, chat_refusal="I can't help with that.")
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError, match="I can't help with that."):
        service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert _generation_history_files(tmp_path) == []


def test_generate_uses_structured_outputs_with_the_applications_own_schema(tmp_path):
    """The model is passed directly as `response_format` (OpenAI
    Structured Outputs), constraining generation to the application's
    own schema rather than a hand-maintained duplicate or a generic
    `{"type": "json_object"}` instruction.
    """
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_AI_RESPONSE)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [call] = fake_client.chat_calls
    assert call["response_format"] is GeneratedTestCasesResponse


def test_generate_still_independently_validates_the_response_as_a_safety_check(tmp_path):
    """Even with Structured Outputs constraining the API call, a
    malformed response must still be rejected by GenerationService's own
    validation — the API-level constraint is a defense-in-depth layer,
    not a replacement for it.
    """
    fake_client = FakeOpenAIClient(chat_response_content=json.dumps({"testCases": [{"requirementId": "REQ-1"}]}))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError):
        service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert _generation_history_files(tmp_path) == []


def test_generate_raises_external_service_error_when_openai_call_fails_and_saves_no_history(tmp_path):
    fake_client = FakeOpenAIClient(chat_fail_with=RuntimeError("simulated API error"))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ExternalServiceError):
        service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert _generation_history_files(tmp_path) == []


def test_generate_raises_external_service_error_on_openai_timeout_and_saves_no_history(tmp_path):
    fake_client = FakeOpenAIClient(chat_fail_with=TimeoutError("simulated timeout"))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ExternalServiceError):
        service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert _generation_history_files(tmp_path) == []


def test_generation_service_module_never_imports_chromadb_directly():
    """Structural guarantee: GenerationService can't query ChromaDB
    directly even by accident, because it never imports the SDK — every
    vector search stays behind RetrievalService.
    """
    import ast
    import inspect

    from app.services import generation_service as generation_service_module

    tree = ast.parse(inspect.getsource(generation_service_module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom | ast.Import)
        for alias in node.names
    }
    assert "chromadb" not in imported_names


def test_generation_service_module_never_imports_redmine():
    """Structural guarantee mirroring the ChromaDB one above:
    `GenerationService` has no way to call Redmine at all, from any
    request shape — a Redmine ticket, when one is supplied, is always
    looked up by the caller (the frontend) beforehand and handed to
    `generate()` as plain strings.
    """
    import ast
    import inspect

    from app.services import generation_service as generation_service_module

    tree = ast.parse(inspect.getsource(generation_service_module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom | ast.Import)
        for alias in node.names
    }
    assert not any("redmine" in name.lower() for name in imported_names)


# --- Optional Redmine ticket ---


def test_generate_succeeds_with_only_a_feature_and_no_redmine_ticket(tmp_path):
    service, _, _, _ = _make_service(tmp_path)

    entry = service.generate(feature="Appointments")

    assert entry.redmine_ticket == ""
    assert len(entry.test_cases) == 1


def test_generate_uses_the_feature_as_the_retrieval_query_when_nothing_else_is_given(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_AI_RESPONSE)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    service.generate(feature="Appointments")

    assert fake_client.calls == [["Appointments"]]


def test_generate_still_works_when_a_redmine_ticket_is_supplied(tmp_path):
    """Regression check: the optional-ticket change must not disturb the
    existing, fully-specified call shape.
    """
    service, _, _, _ = _make_service(tmp_path)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert entry.redmine_ticket == "12345"


def test_generate_treats_none_redmine_description_as_empty_in_the_persisted_prompt(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_VALID_AI_RESPONSE)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345")

    assert "No Redmine ticket description was provided" in entry.prompt


# --- MAX_GENERATED_TEST_CASES enforcement ---


def _multi_test_case_response(count: int, coverage_note: str | None = None) -> str:
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


def test_generate_truncates_test_cases_exceeding_the_configured_maximum(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_multi_test_case_response(5))
    service, _, _, _ = _make_service(tmp_path, fake_client, max_generated_test_cases=2)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert len(entry.test_cases) == 2
    assert [tc.test_case_id for tc in entry.test_cases] == ["TC-1", "TC-2"]
    assert entry.metadata.generated_test_cases == 2
    # A model overshoot still counts as "target met" once truncated back
    # down to exactly what was requested.
    assert entry.metadata.requested_test_cases == 2
    assert entry.metadata.count_target_met is True


def test_generate_keeps_every_test_case_when_within_the_configured_maximum(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_multi_test_case_response(3))
    service, _, _, _ = _make_service(tmp_path, fake_client, max_generated_test_cases=5)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert len(entry.test_cases) == 3
    assert entry.metadata.requested_test_cases == 5
    assert entry.metadata.count_target_met is False


def test_generate_returns_exactly_five_when_five_are_configured_and_supported(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_multi_test_case_response(5))
    service, _, _, _ = _make_service(tmp_path, fake_client, max_generated_test_cases=5)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert len(entry.test_cases) == 5
    assert entry.metadata.generated_test_cases == 5
    assert entry.metadata.requested_test_cases == 5
    assert entry.metadata.count_target_met is True
    assert entry.metadata.coverage_note is None


def test_generate_returns_exactly_ten_when_ten_are_configured_and_supported(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_multi_test_case_response(10))
    service, _, _, _ = _make_service(tmp_path, fake_client, max_generated_test_cases=10)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert len(entry.test_cases) == 10
    assert entry.metadata.generated_test_cases == 10
    assert entry.metadata.requested_test_cases == 10
    assert entry.metadata.count_target_met is True


def test_generate_records_the_models_reason_when_it_returns_fewer_than_requested(tmp_path, caplog):
    reason = "Only 4 genuinely distinct scenarios are supported by the retrieved workflow context."
    fake_client = FakeOpenAIClient(chat_response_content=_multi_test_case_response(4, coverage_note=reason))
    service, _, _, _ = _make_service(tmp_path, fake_client, max_generated_test_cases=10)

    with caplog.at_level("WARNING"):
        entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert len(entry.test_cases) == 4
    assert entry.metadata.generated_test_cases == 4
    assert entry.metadata.requested_test_cases == 10
    assert entry.metadata.count_target_met is False
    assert entry.metadata.coverage_note == reason
    # "Record/log the reason": the shortfall and the model's own stated
    # reason both land in the application log, not just the returned data.
    assert any(reason in record.message for record in caplog.records)


def test_generate_never_pads_a_short_response_with_fabricated_or_duplicate_test_cases(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_multi_test_case_response(3))
    service, _, _, _ = _make_service(tmp_path, fake_client, max_generated_test_cases=10)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    # Never padded up toward the configured maximum with copies.
    assert len(entry.test_cases) == 3
    ids = [tc.test_case_id for tc in entry.test_cases]
    assert ids == ["TC-1", "TC-2", "TC-3"]
    assert len(ids) == len(set(ids))


def test_generate_sends_the_configured_exact_count_instruction_to_openai(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_multi_test_case_response(10))
    service, _, _, _ = _make_service(tmp_path, fake_client, max_generated_test_cases=10)

    service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    sent_prompt = fake_client.chat_calls[0]["messages"][0]["content"]
    assert "Generate exactly 10 test cases." in sent_prompt


def test_generate_defaults_the_max_test_cases_to_five_when_not_configured(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_multi_test_case_response(8))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert len(entry.test_cases) == 5


# --- Generation progress tracking ---


def test_get_progress_returns_none_for_an_unknown_request_id(tmp_path):
    service, _, _, _ = _make_service(tmp_path)

    assert service.get_progress("never-started") is None


def test_get_progress_is_cleared_once_generation_finishes_successfully(tmp_path):
    service, _, _, _ = _make_service(tmp_path)

    service.generate(
        feature="Appointments", redmine_id="12345", redmine_description="Description text.", request_id="req-1"
    )

    assert service.get_progress("req-1") is None


def test_get_progress_is_cleared_even_when_generation_fails(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content="not valid json")
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError):
        service.generate(feature="Appointments", redmine_id="12345", redmine_description="x", request_id="req-2")

    assert service.get_progress("req-2") is None


def test_generate_progresses_through_every_stage_in_order_when_a_request_id_is_given(tmp_path):
    from app.services import generation_service as generation_service_module

    service, _, _, _ = _make_service(tmp_path)
    observed_stages: list[str] = []
    original_set_stage = service._set_stage  # noqa: SLF001 - intentionally spying on the internal hook

    def _recording_set_stage(request_id, stage):
        observed_stages.append(stage)
        original_set_stage(request_id, stage)

    service._set_stage = _recording_set_stage  # noqa: SLF001

    service.generate(
        feature="Appointments", redmine_id="12345", redmine_description="Description text.", request_id="req-3"
    )

    assert observed_stages == [
        generation_service_module.STAGE_RETRIEVING_DOCUMENTS,
        generation_service_module.STAGE_BUILDING_PROMPT,
        generation_service_module.STAGE_GENERATING_TEST_CASES,
        generation_service_module.STAGE_VALIDATING_RESPONSE,
        generation_service_module.STAGE_SAVING_HISTORY,
    ]


def test_generate_without_a_request_id_never_records_any_progress(tmp_path):
    service, _, _, _ = _make_service(tmp_path)

    service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    # No request_id was given, so there's nothing to have tracked —
    # confirmed indirectly: an unrelated id was never touched either.
    assert service.get_progress("anything") is None


# --- ADO Test Case fields (GenerationService._apply_ado_fields) ---


def _ai_test_case(**overrides) -> dict:
    base = {
        "requirementId": "REQ-1",
        "testCaseId": "TC-1",
        "testCaseTitle": "Reschedule an appointment",
        "priority": "High",
        "testSuite": "Appointments",
        "preconditions": "User is logged in.",
        "steps": [{"stepNo": 1, "action": "Open appointment.", "expectedResult": "Details are shown."}],
        "postConditions": "Appointment is updated.",
        "testType": "Functional",
        "testClassification": "Smoke",
        # The model is instructed to always return null for these — the
        # fixture mirrors that, so these tests prove `GenerationService`
        # supplies them itself rather than merely passing through
        # whatever the model happened to send.
        "workItemType": None,
        "automationStatus": None,
        "state": None,
        "areaPath": None,
        "assignedTo": None,
        "tags": None,
    }
    base.update(overrides)
    return base


def _ai_response(test_cases: list[dict], coverage_note: str | None = None) -> str:
    return json.dumps({"testCases": test_cases, "coverageNote": coverage_note})


def test_generate_gives_a_test_case_with_one_step_exactly_one_step(tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_ai_response([_ai_test_case(steps=[{"stepNo": 1, "action": "a", "expectedResult": "r"}])])
    )
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [test_case] = entry.test_cases
    assert len(test_case.steps) == 1


def test_generate_keeps_multiple_steps_under_one_test_case_never_splitting_them_apart(tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_ai_response(
            [
                _ai_test_case(
                    testCaseTitle="Verify code search in coding tool",
                    steps=[
                        {"stepNo": 1, "action": "Navigate to a patient", "expectedResult": "Patient found"},
                        {"stepNo": 2, "action": "Click on Pencil icon", "expectedResult": "Form should be opened"},
                        {"stepNo": 3, "action": "Search in the code field", "expectedResult": "Code found"},
                        {"stepNo": 4, "action": "Discard the form", "expectedResult": "Form should be discarded"},
                    ],
                )
            ]
        )
    )
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    # One test case, four steps — never four separate test cases.
    [test_case] = entry.test_cases
    assert len(test_case.steps) == 4
    assert [step.step_no for step in test_case.steps] == [1, 2, 3, 4]


def test_generate_restarts_step_numbering_at_one_for_each_independent_test_case(tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_ai_response(
            [
                _ai_test_case(
                    testCaseId="TC-1",
                    steps=[
                        {"stepNo": 1, "action": "Click affiliated provider field", "expectedResult": "Box shown"},
                        {"stepNo": 2, "action": "Search provider", "expectedResult": "Provider found"},
                    ],
                ),
                _ai_test_case(
                    testCaseId="TC-2",
                    steps=[
                        {"stepNo": 1, "action": "Navigate to a patient", "expectedResult": "Patient found"},
                        {"stepNo": 2, "action": "Click Pencil icon", "expectedResult": "Form opened"},
                        {"stepNo": 3, "action": "Discard the form", "expectedResult": "Form discarded"},
                    ],
                ),
            ]
        )
    )
    service, _, _, _ = _make_service(tmp_path, fake_client, max_generated_test_cases=2)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [first, second] = entry.test_cases
    assert [step.step_no for step in first.steps] == [1, 2]
    assert [step.step_no for step in second.steps] == [1, 2, 3]


def test_generate_tags_a_smoke_test_case_with_feature_regression_and_smoke(tmp_path):
    """Regression is the complete suite; Smoke is a subset of it — so a
    Smoke test case carries both tags, never Smoke alone.
    """
    fake_client = FakeOpenAIClient(chat_response_content=_ai_response([_ai_test_case(testClassification="Smoke")]))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Contact and Sticket Log", redmine_id="12345", redmine_description="Description.")

    [test_case] = entry.test_cases
    assert test_case.tags == ["Contact and Sticket Log", "Regression", "Smoke"]


def test_generate_tags_a_regression_only_test_case_with_feature_and_regression(tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_ai_response([_ai_test_case(testClassification="Regression")])
    )
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Contact and Sticket Log", redmine_id="12345", redmine_description="Description.")

    [test_case] = entry.test_cases
    assert test_case.tags == ["Contact and Sticket Log", "Regression"]


def test_generate_never_produces_smoke_without_regression(tmp_path):
    """The exact invariant this change enforces: "Feature; Smoke" with
    no "Regression" must never appear, on any test case, ever.
    """
    fake_client = FakeOpenAIClient(chat_response_content=_ai_response([_ai_test_case(testClassification="Smoke")]))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Contact and Sticket Log", redmine_id="12345", redmine_description="Description.")

    [test_case] = entry.test_cases
    assert test_case.tags != ["Contact and Sticket Log", "Smoke"]
    assert "Regression" in test_case.tags


def test_generate_every_test_case_contains_regression_regardless_of_classification(tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_ai_response(
            [
                _ai_test_case(testCaseId="TC-1", testClassification="Smoke"),
                _ai_test_case(testCaseId="TC-2", testClassification="Regression"),
            ]
        )
    )
    service, _, _, _ = _make_service(tmp_path, fake_client, max_generated_test_cases=2)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    assert all("Regression" in test_case.tags for test_case in entry.test_cases)


def test_generate_does_not_add_smoke_to_every_test_case_automatically(tmp_path):
    """Smoke must stay opt-in per test case — a Regression-classified
    test case never picks up "Smoke" just because another test case in
    the same generation is Smoke.
    """
    fake_client = FakeOpenAIClient(
        chat_response_content=_ai_response(
            [
                _ai_test_case(testCaseId="TC-1", testClassification="Smoke"),
                _ai_test_case(testCaseId="TC-2", testClassification="Regression"),
            ]
        )
    )
    service, _, _, _ = _make_service(tmp_path, fake_client, max_generated_test_cases=2)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [smoke_case, regression_case] = entry.test_cases
    assert "Smoke" in smoke_case.tags
    assert "Smoke" not in regression_case.tags


def test_generate_keeps_the_feature_tag_correct_for_every_test_case(tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_ai_response(
            [
                _ai_test_case(testCaseId="TC-1", testClassification="Smoke"),
                _ai_test_case(testCaseId="TC-2", testClassification="Regression"),
            ]
        )
    )
    service, _, _, _ = _make_service(tmp_path, fake_client, max_generated_test_cases=2)

    entry = service.generate(feature="Service Model 1", redmine_id="12345", redmine_description="Description text.")

    assert all(test_case.tags[0] == "Service Model 1" for test_case in entry.test_cases)


def test_generate_never_invents_a_fourth_tag(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_ai_response([_ai_test_case(testClassification="Smoke")]))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Service Model 1", redmine_id="12345", redmine_description="Description text.")

    [test_case] = entry.test_cases
    assert len(test_case.tags) == 3


def test_generate_always_sets_automation_status_to_not_automated_regardless_of_what_the_model_returns(tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_ai_response([_ai_test_case(automationStatus="Automated")])
    )
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [test_case] = entry.test_cases
    assert test_case.automation_status == "Not Automated"


def test_generate_never_produces_no_as_automation_status(tmp_path):
    """The exact invariant this terminology change enforces: the old
    value ("No") must never appear again, even if the model itself
    returns it verbatim.
    """
    fake_client = FakeOpenAIClient(chat_response_content=_ai_response([_ai_test_case(automationStatus="No")]))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [test_case] = entry.test_cases
    assert test_case.automation_status != "No"
    assert test_case.automation_status == "Not Automated"


def test_generate_automation_status_is_exactly_not_automated_with_correct_casing(tmp_path):
    """Guards against near-miss values the task explicitly rules out:
    "NOT AUTOMATED", "Not automated", "Automated", "Yes"."""
    fake_client = FakeOpenAIClient(chat_response_content=_ai_response([_ai_test_case()]))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [test_case] = entry.test_cases
    assert test_case.automation_status == "Not Automated"
    assert test_case.automation_status not in {"No", "NOT AUTOMATED", "Not automated", "Automated", "Yes"}


def test_generate_always_sets_state_to_design_regardless_of_what_the_model_returns(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_ai_response([_ai_test_case(state="Closed")]))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [test_case] = entry.test_cases
    assert test_case.state == "Design"


def test_generate_always_sets_work_item_type_to_test_case(tmp_path):
    fake_client = FakeOpenAIClient(chat_response_content=_ai_response([_ai_test_case()]))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [test_case] = entry.test_cases
    assert test_case.work_item_type == "Test Case"


def test_generate_never_fabricates_an_area_path_regardless_of_what_the_model_returns(tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_ai_response([_ai_test_case(areaPath="MyProject\\Sandbox")])
    )
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [test_case] = entry.test_cases
    assert test_case.area_path == ""


def test_generate_never_fabricates_an_assignee_regardless_of_what_the_model_returns(tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_ai_response([_ai_test_case(assignedTo="someone@example.com")])
    )
    service, _, _, _ = _make_service(tmp_path, fake_client)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [test_case] = entry.test_cases
    assert test_case.assigned_to == ""


def test_generate_keeps_multiple_test_cases_independent_with_their_own_classification(tmp_path):
    fake_client = FakeOpenAIClient(
        chat_response_content=_ai_response(
            [
                _ai_test_case(testCaseId="TC-1", testClassification="Smoke"),
                _ai_test_case(testCaseId="TC-2", testClassification="Regression"),
            ]
        )
    )
    service, _, _, _ = _make_service(tmp_path, fake_client, max_generated_test_cases=2)

    entry = service.generate(feature="Appointments", redmine_id="12345", redmine_description="Description text.")

    [first, second] = entry.test_cases
    assert first.tags == ["Appointments", "Regression", "Smoke"]
    assert second.tags == ["Appointments", "Regression"]
