"""Orchestrates the full Knowledge Assistant pipeline:

    generate generation_id
    -> RetrievalService -> KnowledgeAssistantPromptBuilder
    -> count tokens (EmbeddingService/tiktoken, no OpenAI call)
    -> persist user_question.json / retrieved_chunks.json / prompt.json
    -> OpenAI Chat Completion (the only OpenAI call this class makes)
    -> validate the response
    -> persist response.json / usage.json

Every step is delegated to an existing, unmodified service — this class
contains no retrieval logic, no prompt formatting, no token-counting or
cost math, and no filesystem logic; it only sequences those services and
persists what they produced through the repository abstraction, never
opening a file itself. `generation_id` is generated once, at the very
start of `ask`, and used for every record persisted for this
request — including `error.json` if any stage fails, so a failed
question is exactly as traceable as a successful one. Never makes a
second OpenAI call for token counting, validation, feedback, retrieval,
or cost — nor does it retry a failed call.

Mirrors `app.services.generation_service.GenerationService`'s OpenAI-
calling shape (Structured Outputs, re-parse-and-revalidate as a safety
check, actual usage always from what OpenAI reports) without importing
from it — that class is the existing, unmodified Test Plan Generator,
answering a different question with a different response schema.
"""
import json
import re
import time
from typing import Any

from openai import OpenAI
from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import ExternalServiceError, ValidationError
from app.core.logging import get_logger
from app.models.generation_record import KnowledgeAssistantErrorStage, RetrievalSourceRecord, RetrievedChunkRef
from app.models.hybrid_retrieval import RetrievalDiagnostics
from app.models.knowledge_assistant_answer import KnowledgeAssistantAnswer
from app.models.knowledge_assistant_prompt import KnowledgeAssistantPromptInput
from app.models.retrieval_result import RetrievalResult, SourceRetrievalResult
from app.models.retrieved_chunk import RetrievedChunk
from app.repositories.generation_repository import GenerationRepository
from app.services.cost_calculator import CostCalculator
from app.services.embedding_service import EmbeddingService
from app.services.knowledge_assistant_prompt_builder import KnowledgeAssistantPromptBuilder
from app.services.retrieval_service import RetrievalService
from app.utils.datetime_utils import utcnow
from app.utils.ids import generate_knowledge_assistant_generation_id

# Redacts anything that looks like an API key/bearer token from an error
# message before it's ever persisted to error.json — a defense-in-depth
# measure, since neither the OpenAI SDK nor this class is expected to
# put one in an exception message in the first place.
_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{10,}|Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE)

logger = get_logger(__name__)


class KnowledgeAssistantService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        prompt_builder: KnowledgeAssistantPromptBuilder,
        embedding_service: EmbeddingService,
        openai_client: OpenAI,
        cost_calculator: CostCalculator,
        generation_repository: GenerationRepository,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._prompt_builder = prompt_builder
        self._embedding_service = embedding_service
        self._openai_client = openai_client
        self._cost_calculator = cost_calculator
        self._generation_repository = generation_repository

    def ask(
        self,
        user_question: str,
        feature: str,
        upload_session_id: str | None = None,
        top_k: int | None = None,
    ) -> tuple[str, KnowledgeAssistantAnswer]:
        """Generates this request's `generation_id` (the caller no
        longer supplies one — this is the sole authority), then
        retrieves context for `user_question` (scoped to `feature`,
        exactly like the Test Plan Generator's retrieval —
        `RetrievalService` has no cross-feature search mode, and this
        reuses it unchanged), builds the Knowledge Assistant prompt,
        persists `user_question.json`/`retrieved_chunks.json`/
        `prompt.json`, then sends that exact prompt to OpenAI Chat
        exactly once. Raises `ExternalServiceError` if the OpenAI call
        itself fails, or `ValidationError` if it succeeds but returns
        something empty, malformed, or citing a document that was never
        retrieved. Returns `(generation_id, answer)` on success.

        A failure at *any* stage persists `error.json` (tagged with
        that stage and the `generation_id`, which stays valid and
        traceable regardless of where the failure happened) before
        re-raising — the original exception always propagates, whether
        or not persisting the error itself succeeds. Neither
        `response.json` nor `usage.json` is ever written unless the
        whole pipeline succeeds, and this method never retries.
        """
        generation_id = generate_knowledge_assistant_generation_id()
        started_at = time.perf_counter()
        stage = KnowledgeAssistantErrorStage.INPUT_VALIDATION

        try:
            if not user_question or not user_question.strip():
                raise ValidationError("The question must not be empty.")

            stage = KnowledgeAssistantErrorStage.PERSISTENCE
            self._generation_repository.save_user_question(generation_id, user_question)

            stage = KnowledgeAssistantErrorStage.RETRIEVAL
            hybrid_result = self._retrieval_service.retrieve_hybrid(
                query_text=user_question,
                feature=feature,
                upload_session_id=upload_session_id,
                hybrid_candidate_chunks=top_k,
            )
            retrieval_result = hybrid_result.retrieval_result

            stage = KnowledgeAssistantErrorStage.PERSISTENCE
            retrieved_document_names = self._persist_retrieved_chunks(generation_id, retrieval_result)

            stage = KnowledgeAssistantErrorStage.PROMPT_BUILDING
            prompt_text, estimated_tokens = self._persist_prompt(generation_id, user_question, retrieval_result)

            stage = KnowledgeAssistantErrorStage.PERSISTENCE
            self._persist_retrieval_diagnostics(generation_id, hybrid_result.diagnostics, estimated_tokens)

            stage = KnowledgeAssistantErrorStage.OPENAI_REQUEST
            completion = self._call_openai(prompt_text)

            stage = KnowledgeAssistantErrorStage.RESPONSE_VALIDATION
            answer = _parse_answer(_extract_response_text(completion))
            _validate_source_documents(answer, retrieved_document_names)
            actual_usage = self._cost_calculator.calculate_actual_usage_from_openai(completion.usage)

            # Diagnostics only — `actual_usage` above (OpenAI's own
            # reported prompt/completion tokens) is what's priced and
            # what `output_tokens` means everywhere else; nothing here
            # ever overrides it. `usage_detail` surfaces
            # `reasoning_tokens`/`cached_tokens` when OpenAI's response
            # includes them (a reasoning model like GPT-5 can spend most
            # of `completion_tokens` on invisible reasoning, never
            # appearing in `answer.answer`); the raw object is logged in
            # full at DEBUG level so the exact source of a surprising
            # `output_tokens` figure is always inspectable, not guessed.
            usage_detail = self._cost_calculator.parse_openai_usage_detail(completion.usage)
            logger.debug("Raw OpenAI usage for generation '%s': %s", generation_id, _usage_to_dict(completion.usage))
            visible_answer_tokens = self._cost_calculator.count_tokens(answer.answer)
            structured_output_overhead_tokens = max(
                0, actual_usage.completion_tokens - (usage_detail.reasoning_tokens or 0) - visible_answer_tokens
            )

            stage = KnowledgeAssistantErrorStage.PERSISTENCE
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            self._generation_repository.save_response(
                generation_id,
                response_answer=answer.answer,
                output_tokens=actual_usage.completion_tokens,
                source_documents=answer.source_documents,
            )
            self._generation_repository.save_usage(
                generation_id,
                model=self._cost_calculator.model,
                input_tokens=actual_usage.prompt_tokens,
                output_tokens=actual_usage.completion_tokens,
                total_tokens=actual_usage.total_tokens,
                estimated_input_tokens=estimated_tokens,
                input_cost_inr=actual_usage.input_cost.inr,
                output_cost_inr=actual_usage.output_cost.inr,
                total_cost_inr=actual_usage.total_cost.inr,
                generation_time_ms=elapsed_ms,
                reasoning_tokens=usage_detail.reasoning_tokens,
                cached_tokens=usage_detail.cached_tokens,
                visible_answer_tokens=visible_answer_tokens,
                structured_output_overhead_tokens=structured_output_overhead_tokens,
            )

            return generation_id, answer
        except Exception as exc:
            self._persist_error(generation_id, exc, stage)
            raise

    def _persist_error(self, generation_id: str, exc: Exception, stage: KnowledgeAssistantErrorStage) -> None:
        """Best-effort: if persisting the error record itself fails
        (e.g. a disk problem), that failure is swallowed rather than
        masking the original exception, which always propagates from
        `ask` regardless.
        """
        try:
            self._generation_repository.save_error(
                generation_id,
                error_type=type(exc).__name__,
                error_message=_redact_secrets(str(exc)),
                stage=stage,
            )
        except Exception:
            pass

    def _call_openai(self, prompt_text: str) -> Any:
        """Uses OpenAI's Structured Outputs (`beta.chat.completions.parse`
        with `response_format=KnowledgeAssistantAnswer`) so the model is
        constrained, at the API level, to emit exactly this schema's
        camelCase property names — not just instructed to via the prompt.
        Sends `prompt_text` exactly as `KnowledgeAssistantPromptBuilder`
        built it: no second prompt constructed here, no rewriting or
        classifying the question first.
        """
        try:
            return self._openai_client.beta.chat.completions.parse(
                model=self._cost_calculator.model,
                messages=[{"role": "user", "content": prompt_text}],
                response_format=KnowledgeAssistantAnswer,
            )
        except Exception as exc:
            raise ExternalServiceError(f"OpenAI chat completion request failed: {exc}") from exc

    def _persist_retrieved_chunks(self, generation_id: str, retrieval_result: RetrievalResult) -> set[str]:
        workflow = _source_record(retrieval_result.workflow)
        historical_test_cases = _source_record(retrieval_result.historical_test_cases)
        historical_issues = _source_record(retrieval_result.historical_issues)
        uploaded_documents = _source_record(retrieval_result.uploaded_documents)

        all_chunks = [
            *workflow.chunks,
            *historical_test_cases.chunks,
            *historical_issues.chunks,
            *uploaded_documents.chunks,
        ]
        all_documents = {
            *workflow.documents,
            *historical_test_cases.documents,
            *historical_issues.documents,
            *uploaded_documents.documents,
        }

        self._generation_repository.save_retrieved_chunks(
            generation_id,
            chunks_retrieved=all_chunks,
            documents_retrieved=sorted(all_documents),
            workflow=workflow,
            historical_test_cases=historical_test_cases,
            historical_issues=historical_issues,
            uploaded_documents=uploaded_documents,
        )
        return all_documents

    def _persist_prompt(
        self, generation_id: str, user_question: str, retrieval_result: RetrievalResult
    ) -> tuple[str, int]:
        prompt_input = KnowledgeAssistantPromptInput(
            user_question=user_question,
            workflow_chunks=retrieval_result.workflow_results,
            historical_test_case_chunks=retrieval_result.test_case_results,
            historical_issue_chunks=retrieval_result.issue_results,
            uploaded_document_chunks=retrieval_result.upload_results,
        )
        prompt_result = self._prompt_builder.build(prompt_input, generated_at=utcnow())

        # `estimated_tokens`: the full prompt as it will actually be
        # sent (fixed system instructions/rules + retrieved context +
        # question + answer instructions) — what cost is billed on.
        # `input_tokens`: just the *dynamic* portion the user/retrieval
        # actually contributed (retrieved context + the question
        # itself), excluding the fixed instructions boilerplate that's
        # identical on every call — a smaller number, useful for
        # telling "your content" apart from "our fixed prompt". Both
        # come from one local tiktoken call — never an OpenAI call.
        dynamic_input_text = "\n\n".join(
            [prompt_result.sections.retrieved_context, prompt_result.sections.user_question_section]
        )
        input_tokens, estimated_tokens = self._embedding_service.count_tokens(
            [dynamic_input_text, prompt_result.prompt_text]
        )

        self._generation_repository.save_prompt(
            generation_id,
            prompt=prompt_result.prompt_text,
            input_tokens=input_tokens,
            estimated_tokens=estimated_tokens,
            prompt_version=prompt_result.prompt_version,
        )
        return prompt_result.prompt_text, estimated_tokens

    def _persist_retrieval_diagnostics(
        self, generation_id: str, diagnostics: RetrievalDiagnostics, estimated_prompt_tokens: int
    ) -> None:
        """`retrieve_hybrid` has no prompt to count tokens for yet at
        the point it runs, so `estimated_prompt_tokens` is filled in
        here, once `_persist_prompt` has actually built one — the only
        field on `RetrievalDiagnostics` not already final when
        retrieval itself finishes.
        """
        self._generation_repository.save_retrieval_diagnostics(
            generation_id, diagnostics.model_copy(update={"estimated_prompt_tokens": estimated_prompt_tokens})
        )


def _redact_secrets(message: str) -> str:
    return _SECRET_PATTERN.sub("[REDACTED]", message)


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    """Renders an OpenAI `usage` object as a plain dict for logging,
    without assuming it's a real pydantic `CompletionUsage` (a test
    double may be a bare `SimpleNamespace`) — `None` only if neither
    shape applies, so a logging call can never itself raise.
    """
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "__dict__"):
        return dict(vars(usage))
    return None


def _extract_response_text(completion: Any) -> str:
    try:
        message = completion.choices[0].message
        content = message.content
    except (AttributeError, IndexError) as exc:
        raise ValidationError(f"OpenAI response did not contain a completion message: {exc}") from exc
    if not content:
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValidationError(f"OpenAI declined to answer the question: {refusal}")
        raise ValidationError("OpenAI returned an empty response.")
    return content


def _parse_answer(raw_text: str) -> KnowledgeAssistantAnswer:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"AI response was not valid JSON: {exc}") from exc

    try:
        answer = KnowledgeAssistantAnswer.model_validate(payload)
    except PydanticValidationError as exc:
        raise ValidationError(f"AI response did not match the expected answer schema: {exc}") from exc

    if not answer.answer.strip():
        raise ValidationError("AI response contained an empty answer.")
    return answer


def _validate_source_documents(answer: KnowledgeAssistantAnswer, retrieved_document_names: set[str]) -> None:
    """The model is never trusted to only cite real documents — Structured
    Outputs constrains the response's *shape*, not whether a cited name
    is truthful, so this checks every `source_documents` entry against
    what was actually retrieved (persisted by `_persist_retrieved_chunks`)
    and rejects the whole response if the model invented one.
    """
    invented = [name for name in answer.source_documents if name not in retrieved_document_names]
    if invented:
        raise ValidationError(
            f"AI response cited source document(s) that were never retrieved: {', '.join(invented)}"
        )


def _source_record(source: SourceRetrievalResult) -> RetrievalSourceRecord:
    chunks = [_chunk_ref(chunk) for chunk in source.merged_chunks]
    documents = sorted({chunk.document_name for chunk in chunks})
    return RetrievalSourceRecord(
        candidate_retrieved=len(source.candidate_chunks),
        final_chunks=len(source.final_chunks),
        merged_chunks=len(source.merged_chunks),
        documents=documents,
        chunks=chunks,
    )


def _chunk_ref(chunk: RetrievedChunk) -> RetrievedChunkRef:
    """Mirrors `app.routers.retrieval._chunk_out`'s merge-diagnostics
    mapping (`merged`/`merged_chunk_range`/`merged_chunk_count`) so the
    same "was this chunk the result of merging adjacent originals"
    information persisted here matches what `/retrieval/debug` already
    reports live — no separate mapping logic invented.
    """
    merged_chunk_numbers = chunk.merged_chunk_numbers
    is_merged = merged_chunk_numbers is not None
    return RetrievedChunkRef(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id or "",
        document_name=chunk.source_filename,
        similarity_score=chunk.similarity_score,
        artifact_type=chunk.artifact_type,
        section_heading=chunk.section_heading,
        page_number=chunk.page_number,
        chunk_number=chunk.chunk_number,
        merged=is_merged,
        merged_chunk_range=(f"{min(merged_chunk_numbers)}–{max(merged_chunk_numbers)}" if is_merged else None),
        merged_chunk_count=(len(merged_chunk_numbers) if is_merged else None),
    )
