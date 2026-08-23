"""Aggregates usage across the two independent AI generation sources —
Test Plan Generator (`backend/data/history/generation/`, via
`HistoryService`) and Knowledge Assistant (`backend/database/generations/`,
via `GenerationRepository`) — into one normalized view for the Usage
Dashboard. Neither source's own persistence, history behavior, or data
format is touched here; this service only reads what each already
exposes and re-shapes it into `NormalizedUsageRecord`.
"""
from app.core.logging import get_logger
from app.models.generation_record import UsageRecord
from app.models.generation_summary import GenerationSummary
from app.models.normalized_usage import NormalizedUsageRecord, UnifiedUsage, UnifiedUsageSummary, UsageSource
from app.repositories.generation_repository import GenerationRepository
from app.services.cost_calculator import CostCalculator
from app.services.history_service import HistoryService

logger = get_logger(__name__)


class UsageService:
    def __init__(
        self,
        history_service: HistoryService,
        generation_repository: GenerationRepository,
        cost_calculator: CostCalculator,
    ) -> None:
        self._history_service = history_service
        self._generation_repository = generation_repository
        self._cost_calculator = cost_calculator

    def get_test_plan_generation_usage(self) -> list[NormalizedUsageRecord]:
        """Test Plan Generator's existing history, normalized — the
        underlying data source and its own corrupted-file handling
        (`HistoryService.list_generations`) are unchanged. If the source
        itself is unavailable for some other reason, this returns an
        empty list rather than taking down the whole dashboard; the
        Knowledge Assistant side is unaffected either way.
        """
        try:
            summaries = self._history_service.list_generation_summaries(self._cost_calculator)
        except Exception:
            logger.warning("Failed to load Test Plan Generator usage history for the Usage Dashboard.", exc_info=True)
            return []
        return [_normalize_test_plan_summary(summary) for summary in summaries]

    def get_knowledge_assistant_usage(self) -> list[NormalizedUsageRecord]:
        """Knowledge Assistant's persisted `usage.json` files, normalized
        — read directly, never derived from `prompt.json`/`response.json`.
        A missing or corrupted individual `usage.json` is already skipped
        and logged by `GenerationRepository.list_usage_records()`; if the
        whole source is unavailable, this returns an empty list rather
        than taking down the whole dashboard.
        """
        try:
            records = self._generation_repository.list_usage_records()
        except Exception:
            logger.warning("Failed to load Knowledge Assistant usage records for the Usage Dashboard.", exc_info=True)
            return []
        return [_normalize_knowledge_assistant_record(record) for record in records]

    def get_usage(self) -> UnifiedUsage:
        generations = self.get_test_plan_generation_usage() + self.get_knowledge_assistant_usage()
        generations.sort(key=lambda record: record.created_at, reverse=True)

        total_generations = len(generations)
        total_tokens = sum(record.total_tokens for record in generations)
        total_ai_cost_inr = sum(record.total_cost_inr for record in generations)
        average_cost_inr = total_ai_cost_inr / total_generations if total_generations else 0.0

        return UnifiedUsage(
            summary=UnifiedUsageSummary(
                total_generations=total_generations,
                total_tokens=total_tokens,
                total_ai_cost_inr=total_ai_cost_inr,
                average_cost_inr=average_cost_inr,
            ),
            generations=generations,
        )


def _normalize_test_plan_summary(summary: GenerationSummary) -> NormalizedUsageRecord:
    """`total_cost_inr` is `summary.total_ai_cost_inr` — that field is
    already `embedding_cost_inr (if any) + actual_generation_cost_inr`,
    preserving the meaning "Total AI Cost" has always had elsewhere in
    this dashboard. `input_cost_inr`/`output_cost_inr` cover only the AI
    Generation stage, so the two will NOT sum to `total_cost_inr` for a
    row with an associated Embedding stage cost — the Knowledge
    Assistant has no such stage, so its rows never have this gap.
    """
    return NormalizedUsageRecord(
        generation_id=summary.generation_id,
        source=UsageSource.TEST_PLAN_GENERATOR,
        model=summary.model,
        input_tokens=summary.prompt_tokens,
        output_tokens=summary.completion_tokens,
        total_tokens=summary.total_tokens,
        estimated_input_tokens=summary.estimated_prompt_tokens,
        input_cost_inr=summary.actual_generation_input_cost_inr,
        output_cost_inr=summary.actual_generation_output_cost_inr,
        total_cost_inr=summary.total_ai_cost_inr,
        generation_time_ms=summary.generation_time_ms,
        created_at=summary.created_at,
    )


def _normalize_knowledge_assistant_record(record: UsageRecord) -> NormalizedUsageRecord:
    return NormalizedUsageRecord(
        generation_id=record.generation_id,
        source=UsageSource.KNOWLEDGE_ASSISTANT,
        model=record.model,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        total_tokens=record.total_tokens,
        estimated_input_tokens=record.estimated_input_tokens,
        input_cost_inr=record.input_cost_inr,
        output_cost_inr=record.output_cost_inr,
        total_cost_inr=record.total_cost_inr,
        generation_time_ms=record.generation_time_ms,
        created_at=record.created_at,
        reasoning_tokens=record.reasoning_tokens,
        cached_tokens=record.cached_tokens,
        visible_answer_tokens=record.visible_answer_tokens,
        structured_output_overhead_tokens=record.structured_output_overhead_tokens,
    )
