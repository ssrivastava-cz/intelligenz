from pydantic import BaseModel


class GenerationStatistics(BaseModel):
    """Dashboard aggregates computed directly from persisted generation
    history by `HistoryService.generation_statistics()` — no separate
    statistics store, no regenerated data.
    """

    total_generations: int
    successful_generations: int
    failed_generations: int
    average_generation_time_ms: float
    average_test_cases: float
    average_prompt_tokens: float
    average_completion_tokens: float
    average_cost_usd: float
    average_cost_inr: float
    most_used_feature: str | None
    most_used_model: str | None
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_spend_usd: float
    total_spend_inr: float
