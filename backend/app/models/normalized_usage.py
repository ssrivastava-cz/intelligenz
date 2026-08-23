"""Unified Usage Dashboard view — normalizes two independent, differently-
shaped usage sources (Test Plan Generator's `GenerationSummary`, from
`backend/data/history/generation/`, and Knowledge Assistant's
`UsageRecord`, from `backend/database/generations/*/usage.json`) into one
common record, so the dashboard can render, sort, and aggregate across
both without knowing which source any given row came from. See
`app.services.usage_service.UsageService` for the normalization itself —
nothing here reads either source; this module only defines the shape
they're both mapped onto.
"""
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class UsageSource(StrEnum):
    TEST_PLAN_GENERATOR = "test_plan_generator"
    KNOWLEDGE_ASSISTANT = "knowledge_assistant"


class NormalizedUsageRecord(BaseModel):
    """One generation's usage/cost, regardless of which feature produced
    it. `total_cost_inr` is each source's own established "Total AI
    Cost" figure — for Knowledge Assistant that's exactly
    `input_cost_inr + output_cost_inr`, but for Test Plan Generator it
    also includes that generation's Embedding-stage cost (if any), so
    the two won't always sum to `total_cost_inr` there. See
    `UsageService._normalize_test_plan_summary`.
    """

    generation_id: str
    source: UsageSource
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_input_tokens: int
    input_cost_inr: float
    output_cost_inr: float
    total_cost_inr: float
    generation_time_ms: float
    created_at: datetime
    # OpenAI's detailed output-token breakdown plus local diagnostics —
    # only ever populated for Knowledge Assistant rows (see
    # `UsageService._normalize_knowledge_assistant_record`); `None` for
    # every Test Plan Generator row, since that source's own usage
    # capture (`CostCalculator.calculate_actual_usage_from_openai`,
    # deliberately unchanged) never collects this detail. `None` here
    # means "not tracked for this source", never "confirmed zero".
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    visible_answer_tokens: int | None = None
    structured_output_overhead_tokens: int | None = None


class UnifiedUsageSummary(BaseModel):
    """Dashboard summary cards, computed across both sources together."""

    total_generations: int
    total_tokens: int
    total_ai_cost_inr: float
    average_cost_inr: float


class UnifiedUsage(BaseModel):
    """`UsageService.get_usage()`'s full result: the combined summary
    plus every normalized record, newest first.
    """

    summary: UnifiedUsageSummary
    generations: list[NormalizedUsageRecord]
