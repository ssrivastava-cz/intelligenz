from pydantic import BaseModel


class MoneyAmount(BaseModel):
    """A cost expressed in both USD (the source of truth — OpenAI bills
    in USD) and INR (a convenience conversion for local reporting). All
    currency conversion happens only inside `CostCalculator`; every
    other component just reads both fields already computed.
    """

    usd: float
    inr: float


class EstimatedUsage(BaseModel):
    """Projected input cost for a prompt that hasn't been sent to OpenAI
    Chat yet — computed locally via `EmbeddingService.count_tokens` and
    `CostCalculator`, never by calling OpenAI.
    """

    model: str
    input_tokens: int
    estimated_input_cost: MoneyAmount
    currency: str = "USD"


class ActualUsage(BaseModel):
    """The real usage OpenAI reported for a completed Chat generation,
    with cost computed from configured pricing (never hardcoded).
    """

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: MoneyAmount
    output_cost: MoneyAmount
    total_cost: MoneyAmount


class TokenUsageDetail(BaseModel):
    """Every token-count field OpenAI's Chat Completions `usage` object
    can report, captured verbatim from `completion.usage` — never
    altered, estimated, or backfilled. `None` for any sub-field the API
    response didn't include (`completion_tokens_details`/
    `prompt_tokens_details` are themselves optional on OpenAI's side —
    some models/responses omit them entirely).

    This exists specifically because `completion_tokens` (this
    project's `output_tokens`) can be dramatically larger than the
    visible answer's own token count for a reasoning model like GPT-5:
    `reasoning_tokens` is billed as output but never appears in the
    response content. See `CostCalculator.parse_openai_usage_detail`.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None
    audio_tokens_prompt: int | None = None
    audio_tokens_completion: int | None = None


class PricingSnapshot(BaseModel):
    """The exact per-token pricing (and USD->INR exchange rate) in
    effect when a generation ran, persisted alongside its cost so
    historical records stay reproducible even if pricing configuration
    or exchange rates change later.
    """

    model: str
    input_price_per_million_tokens: float
    output_price_per_million_tokens: float
    usd_to_inr_exchange_rate: float
