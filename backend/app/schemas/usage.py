from datetime import datetime

from app.models.normalized_usage import UsageSource
from app.schemas.common import CamelModel


class NormalizedUsageRecordOut(CamelModel):
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
    # `None` for every Test Plan Generator row (not tracked for that
    # source); see `app.models.normalized_usage.NormalizedUsageRecord`.
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    visible_answer_tokens: int | None = None
    structured_output_overhead_tokens: int | None = None


class UsageSummaryOut(CamelModel):
    total_generations: int
    total_tokens: int
    total_ai_cost_inr: float
    average_cost_inr: float


class UsageResponse(CamelModel):
    summary: UsageSummaryOut
    generations: list[NormalizedUsageRecordOut]
