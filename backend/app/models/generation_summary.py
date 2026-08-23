from datetime import datetime

from pydantic import BaseModel

from app.models.cost import MoneyAmount
from app.models.generation_history import GenerationHistoryEntry


class GenerationSummary(BaseModel):
    """The trimmed, list-view projection of one `GenerationHistoryEntry`
    — used by the `/generation/history` listing endpoint, the Usage
    Dashboard, and `HistoryService.generation_statistics()` — so this
    field mapping (flattening nested usage/cost fields, deriving
    counts) exists in exactly one place.

    Surfaces all three cost stages of a generation's lifecycle:
    Embedding (looked up from the matching Upload Embedding history by
    `HistoryService`, since a `GenerationHistoryEntry` only stores the
    `upload_session_id`, not the embedding cost itself), Prompt
    Construction (`estimated_*` — the pre-generation estimate), and AI
    Generation (`prompt_tokens`/`completion_tokens`/
    `actual_generation_cost_*` — OpenAI's actual reported usage).
    `total_ai_cost_*` is exactly `embedding_cost_* + actual_generation_cost_*`
    (0 for the embedding side when a generation has no associated
    upload session or that session was never embedded).
    """

    generation_id: str
    created_at: datetime
    feature: str
    redmine_ticket: str
    model: str
    prompt_version: str
    generation_time_ms: float
    number_of_test_cases: int

    # AI Generation stage — OpenAI's actual reported usage, never estimated.
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    actual_generation_cost_usd: float
    actual_generation_cost_inr: float
    # The same actual cost, split into its input/output halves (both
    # already present on `entry.actual_usage`, just not previously
    # surfaced here) — added for `UsageService`'s cross-source
    # normalization, which needs an input/output split the same way
    # the Knowledge Assistant's `UsageRecord` already has one. Never
    # persisted to disk; purely an in-memory projection, exactly like
    # every other field on this class.
    actual_generation_input_cost_inr: float
    actual_generation_output_cost_inr: float

    # Prompt Construction stage — the pre-generation estimate.
    estimated_prompt_tokens: int
    estimated_input_cost_usd: float
    estimated_input_cost_inr: float

    # Embedding stage — from the Upload Embedding pipeline, keyed by
    # upload session. None when this generation has no associated
    # upload session, or that session was never embedded.
    embedding_tokens: int | None
    embedding_cost_usd: float | None
    embedding_cost_inr: float | None

    # Total AI Cost — embedding_cost + actual_generation_cost, exactly.
    total_ai_cost_usd: float
    total_ai_cost_inr: float

    status: str

    @classmethod
    def from_entry(
        cls,
        entry: GenerationHistoryEntry,
        embedding_tokens: int | None = None,
        embedding_cost: MoneyAmount | None = None,
    ) -> "GenerationSummary":
        actual_generation_cost = entry.actual_usage.total_cost
        embedding_cost_usd = embedding_cost.usd if embedding_cost else None
        embedding_cost_inr = embedding_cost.inr if embedding_cost else None
        return cls(
            generation_id=entry.generation_id,
            created_at=entry.timestamp,
            feature=entry.feature,
            redmine_ticket=entry.redmine_ticket,
            model=entry.model,
            prompt_version=entry.prompt_version,
            generation_time_ms=entry.generation_time_ms,
            number_of_test_cases=len(entry.test_cases),
            prompt_tokens=entry.actual_usage.prompt_tokens,
            completion_tokens=entry.actual_usage.completion_tokens,
            total_tokens=entry.actual_usage.total_tokens,
            actual_generation_cost_usd=actual_generation_cost.usd,
            actual_generation_cost_inr=actual_generation_cost.inr,
            actual_generation_input_cost_inr=entry.actual_usage.input_cost.inr,
            actual_generation_output_cost_inr=entry.actual_usage.output_cost.inr,
            estimated_prompt_tokens=entry.estimated_usage.input_tokens,
            estimated_input_cost_usd=entry.estimated_usage.estimated_input_cost.usd,
            estimated_input_cost_inr=entry.estimated_usage.estimated_input_cost.inr,
            embedding_tokens=embedding_tokens,
            embedding_cost_usd=embedding_cost_usd,
            embedding_cost_inr=embedding_cost_inr,
            total_ai_cost_usd=(embedding_cost_usd or 0.0) + actual_generation_cost.usd,
            total_ai_cost_inr=(embedding_cost_inr or 0.0) + actual_generation_cost.inr,
            status=entry.status,
        )
