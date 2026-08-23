"""API-facing cost/usage schemas, shared by the Prompt Builder debug
endpoint and the Generation debug endpoint so both present identical
shapes for the identical underlying `CostCalculator` output.
"""
from app.schemas.common import CamelModel


class MoneyAmountOut(CamelModel):
    usd: float
    inr: float


class EstimatedUsageOut(CamelModel):
    model: str
    input_tokens: int
    estimated_input_cost: MoneyAmountOut
    currency: str


class ActualUsageOut(CamelModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: MoneyAmountOut
    output_cost: MoneyAmountOut
    total_cost: MoneyAmountOut


class PricingSnapshotOut(CamelModel):
    model: str
    input_price_per_million_tokens: float
    output_price_per_million_tokens: float
    usd_to_inr_exchange_rate: float
