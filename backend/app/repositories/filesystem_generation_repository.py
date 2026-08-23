"""Filesystem-backed `GenerationRepository` — the only implementation
today. Stores one directory per generation under
`<database_root>/generations/<generation_id>/`, holding one JSON file
per record type (`user_question.json`, `retrieved_chunks.json`,
`prompt.json`, `response.json`, `feedback.json`, `usage.json`).

Deliberately unaware of any business logic (retrieval, prompt building,
cost calculation) — it only persists exactly what it's given, correlated
by the `generation_id` the caller supplies, and reads it back. When a
future `MongoGenerationRepository` replaces this, no caller of
`GenerationRepository` needs to change, since both satisfy the same
abstract interface.
"""
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import ExternalServiceError, NotFoundError
from app.core.logging import get_logger
from app.models.generation_record import (
    FeedbackRecord,
    GenerationErrorRecord,
    GenerationFeedbackEvaluation,
    GenerationListItem,
    GenerationRecord,
    KnowledgeAssistantErrorStage,
    PromptRecord,
    ResponseRecord,
    RetrievalDiagnosticsRecord,
    RetrievalSourceRecord,
    RetrievedChunkRef,
    RetrievedChunksRecord,
    UsageRecord,
    UserQuestionRecord,
)
from app.models.hybrid_retrieval import RetrievalDiagnostics
from app.repositories.generation_repository import GenerationRepository
from app.utils.atomic_json import write_json_atomic
from app.utils.datetime_utils import utcnow

_USER_QUESTION_FILENAME = "user_question.json"
_RETRIEVED_CHUNKS_FILENAME = "retrieved_chunks.json"
_RETRIEVAL_DIAGNOSTICS_FILENAME = "retrieval_diagnostics.json"
_PROMPT_FILENAME = "prompt.json"
_RESPONSE_FILENAME = "response.json"
_FEEDBACK_FILENAME = "feedback.json"
_USAGE_FILENAME = "usage.json"
_ERROR_FILENAME = "error.json"

_RecordT = TypeVar("_RecordT", bound=BaseModel)

logger = get_logger(__name__)


class FileSystemGenerationRepository(GenerationRepository):
    def __init__(self, database_root: Path) -> None:
        self._generations_root = database_root / "generations"

    def create_generation(self, generation_id: str) -> None:
        self._generation_dir(generation_id).mkdir(parents=True, exist_ok=True)

    def save_user_question(self, generation_id: str, user_question: str) -> UserQuestionRecord:
        record = UserQuestionRecord(generation_id=generation_id, user_question=user_question, created_at=utcnow())
        self._write(generation_id, _USER_QUESTION_FILENAME, record)
        return record

    def save_retrieved_chunks(
        self,
        generation_id: str,
        chunks_retrieved: list[RetrievedChunkRef],
        documents_retrieved: list[str],
        workflow: RetrievalSourceRecord | None = None,
        historical_test_cases: RetrievalSourceRecord | None = None,
        historical_issues: RetrievalSourceRecord | None = None,
        uploaded_documents: RetrievalSourceRecord | None = None,
    ) -> RetrievedChunksRecord:
        record = RetrievedChunksRecord(
            generation_id=generation_id,
            chunks_retrieved=chunks_retrieved,
            documents_retrieved=documents_retrieved,
            workflow=workflow,
            historical_test_cases=historical_test_cases,
            historical_issues=historical_issues,
            uploaded_documents=uploaded_documents,
            created_at=utcnow(),
        )
        self._write(generation_id, _RETRIEVED_CHUNKS_FILENAME, record)
        return record

    def save_retrieval_diagnostics(
        self, generation_id: str, diagnostics: RetrievalDiagnostics
    ) -> RetrievalDiagnosticsRecord:
        record = RetrievalDiagnosticsRecord(generation_id=generation_id, diagnostics=diagnostics, created_at=utcnow())
        self._write(generation_id, _RETRIEVAL_DIAGNOSTICS_FILENAME, record)
        return record

    def save_prompt(
        self,
        generation_id: str,
        prompt: str,
        input_tokens: int,
        estimated_tokens: int,
        prompt_version: str,
    ) -> PromptRecord:
        record = PromptRecord(
            generation_id=generation_id,
            prompt=prompt,
            input_tokens=input_tokens,
            estimated_tokens=estimated_tokens,
            prompt_version=prompt_version,
            created_at=utcnow(),
        )
        self._write(generation_id, _PROMPT_FILENAME, record)
        return record

    def save_response(
        self, generation_id: str, response_answer: str, output_tokens: int, source_documents: list[str] | None = None
    ) -> ResponseRecord:
        record = ResponseRecord(
            generation_id=generation_id,
            response_answer=response_answer,
            output_tokens=output_tokens,
            source_documents=source_documents or [],
            created_at=utcnow(),
        )
        self._write(generation_id, _RESPONSE_FILENAME, record)
        return record

    def save_feedback(
        self,
        generation_id: str,
        evaluation: GenerationFeedbackEvaluation,
        reason: str | None = None,
        description: str | None = None,
    ) -> FeedbackRecord:
        record = FeedbackRecord(
            generation_id=generation_id,
            evaluation=evaluation,
            reason=reason,
            description=description,
            created_at=utcnow(),
        )
        self._write(generation_id, _FEEDBACK_FILENAME, record)
        return record

    def save_usage(
        self,
        generation_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        estimated_input_tokens: int,
        input_cost_inr: float,
        output_cost_inr: float,
        total_cost_inr: float,
        generation_time_ms: float,
        reasoning_tokens: int | None = None,
        cached_tokens: int | None = None,
        visible_answer_tokens: int | None = None,
        structured_output_overhead_tokens: int | None = None,
    ) -> UsageRecord:
        record = UsageRecord(
            generation_id=generation_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_input_tokens=estimated_input_tokens,
            input_cost_inr=input_cost_inr,
            output_cost_inr=output_cost_inr,
            total_cost_inr=total_cost_inr,
            generation_time_ms=generation_time_ms,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            visible_answer_tokens=visible_answer_tokens,
            structured_output_overhead_tokens=structured_output_overhead_tokens,
            created_at=utcnow(),
        )
        self._write(generation_id, _USAGE_FILENAME, record)
        return record

    def save_error(
        self,
        generation_id: str,
        error_type: str,
        error_message: str,
        stage: KnowledgeAssistantErrorStage,
    ) -> GenerationErrorRecord:
        record = GenerationErrorRecord(
            generation_id=generation_id,
            error_type=error_type,
            error_message=error_message,
            stage=stage,
            created_at=utcnow(),
        )
        self._write(generation_id, _ERROR_FILENAME, record)
        return record

    def get_generation(self, generation_id: str) -> GenerationRecord:
        generation_dir = self._generation_dir(generation_id)
        if not generation_dir.is_dir():
            raise NotFoundError(f"No generation found with id '{generation_id}'.")

        return GenerationRecord(
            generation_id=generation_id,
            user_question=self._read_strict(generation_dir, _USER_QUESTION_FILENAME, UserQuestionRecord),
            retrieved_chunks=self._read_strict(generation_dir, _RETRIEVED_CHUNKS_FILENAME, RetrievedChunksRecord),
            retrieval_diagnostics=self._read_strict(
                generation_dir, _RETRIEVAL_DIAGNOSTICS_FILENAME, RetrievalDiagnosticsRecord
            ),
            prompt=self._read_strict(generation_dir, _PROMPT_FILENAME, PromptRecord),
            response=self._read_strict(generation_dir, _RESPONSE_FILENAME, ResponseRecord),
            feedback=self._read_strict(generation_dir, _FEEDBACK_FILENAME, FeedbackRecord),
            usage=self._read_strict(generation_dir, _USAGE_FILENAME, UsageRecord),
            error=self._read_strict(generation_dir, _ERROR_FILENAME, GenerationErrorRecord),
        )

    def list_generations(self) -> list[GenerationListItem]:
        """A directory whose `user_question.json` is missing or
        corrupted is still included (with `user_question`/`created_at`
        as `None`) rather than skipped outright or failing the whole
        listing — one bad historical record must never break this.
        """
        if not self._generations_root.is_dir():
            return []

        items = []
        for generation_dir in sorted(self._generations_root.iterdir()):
            if not generation_dir.is_dir():
                continue
            question = self._read_lenient(generation_dir, _USER_QUESTION_FILENAME, UserQuestionRecord)
            items.append(
                GenerationListItem(
                    generation_id=generation_dir.name,
                    user_question=question.user_question if question else None,
                    created_at=question.created_at if question else None,
                )
            )

        # Newest first among items with a known `created_at`; anything
        # missing one (an unreadable `user_question.json`) sorts last,
        # rather than being excluded.
        items.sort(key=lambda item: (item.created_at is None, -item.created_at.timestamp() if item.created_at else 0.0))
        return items

    def list_usage_records(self) -> list[UsageRecord]:
        if not self._generations_root.is_dir():
            return []

        records = []
        for generation_dir in sorted(self._generations_root.iterdir()):
            if not generation_dir.is_dir():
                continue
            try:
                usage = self._read_strict(generation_dir, _USAGE_FILENAME, UsageRecord)
            except ExternalServiceError:
                logger.warning(
                    "Skipping corrupted usage.json for generation '%s' while listing usage records.",
                    generation_dir.name,
                )
                continue
            if usage is not None:
                records.append(usage)
        return records

    def _generation_dir(self, generation_id: str) -> Path:
        return self._generations_root / generation_id

    def _write(self, generation_id: str, filename: str, record: BaseModel) -> None:
        write_json_atomic(self._generation_dir(generation_id) / filename, record.model_dump(mode="json"))

    def _read_strict(self, generation_dir: Path, filename: str, model_cls: type[_RecordT]) -> _RecordT | None:
        """`None` if the file simply doesn't exist yet (not an error —
        e.g. no feedback submitted yet); raises `ExternalServiceError`
        if it exists but is corrupted, since that's a real data-
        integrity problem worth surfacing to whoever asked for this one
        specific generation.
        """
        path = generation_dir / filename
        if not path.is_file():
            return None
        try:
            return model_cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, PydanticValidationError) as exc:
            raise ExternalServiceError(
                f"Generation record '{filename}' for '{generation_dir.name}' is corrupted: {exc}"
            ) from exc

    def _read_lenient(self, generation_dir: Path, filename: str, model_cls: type[_RecordT]) -> _RecordT | None:
        """Like `_read_strict`, but also swallows a corrupted file as
        `None` instead of raising — for `list_generations()`, where one
        bad record must not take down the whole listing.
        """
        try:
            return self._read_strict(generation_dir, filename, model_cls)
        except ExternalServiceError:
            return None
