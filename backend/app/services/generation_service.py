"""Orchestrates the AI Test Case Generation pipeline:

    RetrievalService -> PromptBuilder -> OpenAI Chat Completion
    -> Validate AI Response -> CostCalculator -> HistoryService
    -> Generated Test Cases

The only component in the app allowed to call OpenAI Chat. Every step
is delegated to an existing, unmodified service — this class contains
no retrieval logic, no prompt formatting, no filesystem logic; it only
sequences those services and reports what happened. Generation history
is only ever saved once retrieval, prompt building, the OpenAI call,
and schema validation of its response have *all* already succeeded —
a failure at any step raises before `HistoryService` is ever reached.

The OpenAI call uses Structured Outputs (`beta.chat.completions.parse`
with `response_format=GeneratedTestCasesResponse`), so the model is
constrained at the API level to the application's own schema — the
same Pydantic model, not a hand-maintained duplicate — rather than
relying on prompt wording alone. The raw response text is still
independently re-parsed and re-validated afterward as a safety check.
"""
import json
import threading
import time
from typing import Any

from openai import OpenAI
from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import ExternalServiceError, ValidationError
from app.core.logging import get_logger
from app.models.generated_test_case import GeneratedTestCase, GeneratedTestCasesResponse
from app.models.generation_history import (
    GenerationHistoryDraft,
    GenerationHistoryEntry,
    GenerationMetadata,
    RetrievalSummary,
)
from app.models.prompt_input import PromptBuilderInput
from app.services.cost_calculator import CostCalculator
from app.services.embedding_service import EmbeddingService
from app.services.history_service import HistoryService
from app.services.prompt_builder import PromptBuilder
from app.services.retrieval_service import RetrievalService
from app.utils.datetime_utils import utcnow

_DEFAULT_MAX_GENERATED_TEST_CASES = 5

# ADO Test Case fields that are never the model's call to make — see
# `GeneratedTestCase`'s docstring and `_apply_ado_fields` below. Fixed
# here, in one place, rather than scattered across every call site that
# builds a fresh generated test case.
_ADO_WORK_ITEM_TYPE = "Test Case"
_ADO_AUTOMATION_STATUS = "Not Automated"
_ADO_STATE = "Design"
_ADO_SMOKE = "Smoke"
_ADO_REGRESSION = "Regression"

logger = get_logger(__name__)

# Stage keys `GET /generate/progress/{request_id}` can report — one
# per real internal step `generate()` actually performs (in order).
# "Fetching Redmine ticket details" isn't listed here: the frontend
# performs that lookup itself, before ever calling `POST /generate`, so
# it already observes that step directly without needing this tracker.
STAGE_RETRIEVING_DOCUMENTS = "retrievingDocuments"
STAGE_BUILDING_PROMPT = "buildingPrompt"
STAGE_GENERATING_TEST_CASES = "generatingTestCases"
STAGE_VALIDATING_RESPONSE = "validatingResponse"
STAGE_SAVING_HISTORY = "savingHistory"


class GenerationService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        prompt_builder: PromptBuilder,
        embedding_service: EmbeddingService,
        openai_client: OpenAI,
        cost_calculator: CostCalculator,
        history_service: HistoryService,
        chat_model: str,
        max_generated_test_cases: int = _DEFAULT_MAX_GENERATED_TEST_CASES,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._prompt_builder = prompt_builder
        self._embedding_service = embedding_service
        self._openai_client = openai_client
        self._cost_calculator = cost_calculator
        self._history_service = history_service
        self._chat_model = chat_model
        self._max_generated_test_cases = max_generated_test_cases
        # In-memory, best-effort progress for whatever `POST /generate`
        # calls are currently in flight, keyed by the caller-supplied
        # `request_id` — never persisted, never a job queue: generation
        # still runs synchronously within its own request, this just
        # makes that request's already-real internal steps observable to
        # a concurrent `GET /generate/progress/{request_id}` poll.
        self._progress_lock = threading.Lock()
        self._progress: dict[str, str] = {}

    def get_progress(self, request_id: str) -> str | None:
        """The current stage for an in-flight generation, or `None` if
        `request_id` is unknown (never started, already finished, or
        never supplied in the first place).
        """
        with self._progress_lock:
            return self._progress.get(request_id)

    def _set_stage(self, request_id: str | None, stage: str) -> None:
        if request_id is None:
            return
        with self._progress_lock:
            self._progress[request_id] = stage

    def _clear_progress(self, request_id: str | None) -> None:
        if request_id is None:
            return
        with self._progress_lock:
            self._progress.pop(request_id, None)

    def generate(
        self,
        feature: str,
        redmine_id: str | None = None,
        redmine_description: str | None = None,
        optional_description: str | None = None,
        upload_session_id: str | None = None,
        top_k: int | None = None,
        request_id: str | None = None,
    ) -> GenerationHistoryEntry:
        """Raises `ExternalServiceError` if the OpenAI call itself fails
        (network error, timeout, API error), or `ValidationError` if it
        succeeds but returns something that isn't valid JSON or doesn't
        match the expected test-case schema. Neither path saves history.

        A Redmine ticket is optional: `redmine_id`/`redmine_description`
        may both be omitted, in which case generation proceeds from the
        selected Feature (and whatever knowledge-base context retrieval
        finds for it) alone. `request_id`, if given, is only ever used to
        key this call's progress in `get_progress` — it's never stored,
        never influences retrieval or generation, and is cleared again as
        soon as this method returns (success or failure).
        """
        started_at = time.perf_counter()
        try:
            redmine_ticket_value = redmine_id or ""
            redmine_description_value = redmine_description or ""
            query_text = optional_description or redmine_description_value or feature

            self._set_stage(request_id, STAGE_RETRIEVING_DOCUMENTS)
            retrieval_result = self._retrieval_service.retrieve(
                query_text=query_text, feature=feature, upload_session_id=upload_session_id, top_k=top_k
            )

            self._set_stage(request_id, STAGE_BUILDING_PROMPT)
            prompt_input = PromptBuilderInput(
                feature=feature,
                redmine_ticket=redmine_ticket_value,
                redmine_description=redmine_description_value,
                user_description=optional_description,
                workflow_chunks=retrieval_result.workflow_results,
                historical_test_case_chunks=retrieval_result.test_case_results,
                historical_issue_chunks=retrieval_result.issue_results,
                uploaded_document_chunks=retrieval_result.upload_results,
            )
            prompt_result = self._prompt_builder.build(prompt_input, generated_at=utcnow())

            # Local token estimate for the prompt about to be sent — the
            # same technique `/prompt/debug` uses, computed here only
            # because a completed generation's history record persists
            # both the pre-generation estimate and OpenAI's own
            # post-generation usage side by side.
            prompt_tokens_estimate = self._embedding_service.count_tokens([prompt_result.prompt_text])[0]
            estimated_usage = self._cost_calculator.estimate_input_usage(prompt_tokens_estimate)

            self._set_stage(request_id, STAGE_GENERATING_TEST_CASES)
            completion = self._call_openai(prompt_result.prompt_text)
            raw_response_text = _extract_response_text(completion)

            self._set_stage(request_id, STAGE_VALIDATING_RESPONSE)
            parsed_response = _parse_ai_response(raw_response_text)
            requested_test_cases = self._max_generated_test_cases
            raw_generated_count = len(parsed_response.test_cases)
            parsed_response = _enforce_max_test_cases(parsed_response, requested_test_cases)
            parsed_response = _apply_ado_fields(parsed_response, feature)
            generated_count = len(parsed_response.test_cases)
            count_target_met = generated_count == requested_test_cases
            if raw_generated_count > requested_test_cases:
                logger.warning(
                    "Feature '%s': model returned %d test case(s), exceeding the requested %d — truncated to %d. "
                    "PromptBuilder already instructs the model not to exceed this count; this is a prompt-"
                    "compliance shortfall, not a configuration issue.",
                    feature,
                    raw_generated_count,
                    requested_test_cases,
                    generated_count,
                )
            elif not count_target_met:
                logger.warning(
                    "Feature '%s': generated %d of %d requested test case(s). Model-reported reason: %s",
                    feature,
                    generated_count,
                    requested_test_cases,
                    parsed_response.coverage_note or "not provided by the model.",
                )

            # Actual cost always comes from what OpenAI reports it used,
            # never from the local estimate above.
            actual_usage = self._cost_calculator.calculate_actual_usage_from_openai(completion.usage)
            elapsed_ms = (time.perf_counter() - started_at) * 1000

            self._set_stage(request_id, STAGE_SAVING_HISTORY)
            draft = GenerationHistoryDraft(
                feature=feature,
                redmine_ticket=redmine_ticket_value,
                model=self._chat_model,
                prompt_version=prompt_result.prompt_version,
                retrieval=RetrievalSummary(
                    workflow_chunks=len(retrieval_result.workflow_results),
                    historical_test_cases=len(retrieval_result.test_case_results),
                    historical_issues=len(retrieval_result.issue_results),
                    uploaded_documents=len(retrieval_result.upload_results),
                ),
                estimated_usage=estimated_usage,
                actual_usage=actual_usage,
                pricing=self._cost_calculator.pricing_snapshot(),
                generation_time_ms=elapsed_ms,
                prompt=prompt_result.prompt_text,
                response=raw_response_text,
                test_cases=parsed_response.test_cases,
                metadata=GenerationMetadata(
                    top_k=top_k,
                    upload_session_id=upload_session_id,
                    generated_test_cases=generated_count,
                    requested_test_cases=requested_test_cases,
                    count_target_met=count_target_met,
                    coverage_note=parsed_response.coverage_note,
                ),
            )
            return self._history_service.save_generation_history(draft)
        finally:
            self._clear_progress(request_id)

    def _call_openai(self, prompt_text: str) -> Any:
        """Uses OpenAI's Structured Outputs (`beta.chat.completions.parse`
        with `response_format=GeneratedTestCasesResponse`) so the model is
        constrained, at the API level, to emit exactly this schema's
        camelCase property names — not just instructed to via the prompt.
        The raw JSON text is still independently re-parsed and
        re-validated afterward (`_extract_response_text`/`_parse_ai_response`
        below) as a safety check; this method never trusts the SDK's own
        `.parsed` result.
        """
        try:
            return self._openai_client.beta.chat.completions.parse(
                model=self._chat_model,
                messages=[{"role": "user", "content": prompt_text}],
                response_format=GeneratedTestCasesResponse,
            )
        except Exception as exc:
            raise ExternalServiceError(f"OpenAI chat completion request failed: {exc}") from exc


def _extract_response_text(completion: Any) -> str:
    try:
        message = completion.choices[0].message
        content = message.content
    except (AttributeError, IndexError) as exc:
        raise ValidationError(f"OpenAI response did not contain a completion message: {exc}") from exc
    if not content:
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValidationError(f"OpenAI declined to generate a response: {refusal}")
        raise ValidationError("OpenAI returned an empty response.")
    return content


def _parse_ai_response(raw_text: str) -> GeneratedTestCasesResponse:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"AI response was not valid JSON: {exc}") from exc

    try:
        return GeneratedTestCasesResponse.model_validate(payload)
    except PydanticValidationError as exc:
        raise ValidationError(f"AI response did not match the expected test case schema: {exc}") from exc


def _enforce_max_test_cases(
    response: GeneratedTestCasesResponse, max_test_cases: int
) -> GeneratedTestCasesResponse:
    """Defense in depth: `PromptBuilder` already instructs the model to
    generate exactly `max_test_cases` (or fewer, if the evidence
    doesn't support that many — see `_build_test_case_generation_instructions`),
    but nothing about OpenAI Structured Outputs guarantees an LLM obeys
    a count expressed only in prose. Truncating here (rather than
    trusting the instruction alone) is what actually enforces
    `len(test_cases) <= max_test_cases`; the result stays a plain
    `GeneratedTestCasesResponse` instance, so nothing downstream needs
    to know this happened. Never pads a short response back up to
    `max_test_cases` — a model returning fewer, genuinely-supported test
    cases is the correct, expected behavior, not something to compensate
    for.
    """
    if len(response.test_cases) <= max_test_cases:
        return response
    return response.model_copy(update={"test_cases": response.test_cases[:max_test_cases]})


def _apply_ado_fields(response: GeneratedTestCasesResponse, feature: str) -> GeneratedTestCasesResponse:
    """Applies every ADO Test Case field the model is never trusted to
    decide (see `GeneratedTestCase`'s docstring) — regardless of what it
    returned for them. `tags` is built from a fixed hierarchy, never
    trusted to the model even indirectly: Regression is the complete
    suite and Smoke is a subset of it, so every test case always gets
    the feature tag plus "Regression", and a Smoke-classified test case
    additionally gets "Smoke" on top — never "Smoke" without
    "Regression". The model only ever influences whether a test case
    qualifies as Smoke at all (via `test_classification`), never the
    tags themselves.
    """
    updated_test_cases = [_apply_ado_fields_to_test_case(test_case, feature) for test_case in response.test_cases]
    return response.model_copy(update={"test_cases": updated_test_cases})


def _apply_ado_fields_to_test_case(test_case: GeneratedTestCase, feature: str) -> GeneratedTestCase:
    is_smoke = test_case.test_classification.strip().casefold() == "smoke"
    classification = _ADO_SMOKE if is_smoke else _ADO_REGRESSION
    # Regression is the complete suite; Smoke is a subset of it, never a
    # separate, parallel category — so "Smoke" always implies
    # "Regression" is present too, never the other way around.
    tags = [feature, _ADO_REGRESSION, _ADO_SMOKE] if is_smoke else [feature, _ADO_REGRESSION]
    return test_case.model_copy(
        update={
            "test_classification": classification,
            "work_item_type": _ADO_WORK_ITEM_TYPE,
            "automation_status": _ADO_AUTOMATION_STATUS,
            "state": _ADO_STATE,
            "area_path": "",
            "assigned_to": "",
            "tags": tags,
        }
    )
