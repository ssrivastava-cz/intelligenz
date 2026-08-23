"""Centralizes every LLM cost calculation — estimating a prompt's input
cost before generation, and computing a completed generation's actual
cost afterward — so the Prompt Builder debug endpoint, the future
Generation Service, Generation History, and the future Usage Dashboard
all use identical logic. Pricing (and the USD->INR exchange rate) is
never hardcoded here or anywhere else: it's supplied at construction
from `Settings`, which reads it from configuration. USD is always the
source of truth (OpenAI bills in USD); INR is a convenience conversion
this class derives from it — no other component performs currency
conversion.
"""
import tiktoken

from app.models.cost import ActualUsage, EstimatedUsage, MoneyAmount, PricingSnapshot, TokenUsageDetail

_INR_DECIMAL_PLACES = 2

# OpenAI's newer chat models are tokenized with this encoding; used as a
# fallback when tiktoken doesn't recognize a model name outright (e.g. a
# model newer than the installed tiktoken version knows about).
_FALLBACK_ENCODING = "o200k_base"


class CostCalculator:
    def __init__(
        self,
        model: str,
        input_price_per_million_tokens: float,
        output_price_per_million_tokens: float,
        usd_to_inr_exchange_rate: float,
        cached_input_price_per_million_tokens: float | None = None,
    ) -> None:
        self._model = model
        self._input_price_per_million_tokens = input_price_per_million_tokens
        self._output_price_per_million_tokens = output_price_per_million_tokens
        self._usd_to_inr_exchange_rate = usd_to_inr_exchange_rate
        self._cached_input_price_per_million_tokens = cached_input_price_per_million_tokens
        self._encoding: tiktoken.Encoding | None = None

    @property
    def model(self) -> str:
        return self._model

    def estimate_input_usage(self, input_tokens: int) -> EstimatedUsage:
        input_cost_usd = self._token_cost(input_tokens, self._input_price_per_million_tokens)
        return EstimatedUsage(
            model=self._model,
            input_tokens=input_tokens,
            estimated_input_cost=self._to_money(input_cost_usd),
        )

    def calculate_actual_usage(self, prompt_tokens: int, completion_tokens: int) -> ActualUsage:
        input_cost_usd = self._token_cost(prompt_tokens, self._input_price_per_million_tokens)
        output_cost_usd = self._token_cost(completion_tokens, self._output_price_per_million_tokens)
        input_cost = self._to_money(input_cost_usd)
        output_cost = self._to_money(output_cost_usd)
        return ActualUsage(
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            # usd: exact sum, full precision, for aggregation. inr: sum
            # of the already-rounded parts (not a fresh usd_total * rate
            # rounding), so input + output == total in INR too, not just
            # in USD.
            total_cost=MoneyAmount(
                usd=input_cost_usd + output_cost_usd,
                inr=round(input_cost.inr + output_cost.inr, _INR_DECIMAL_PLACES),
            ),
        )

    def calculate_actual_usage_from_openai(self, usage: object) -> ActualUsage:
        """Parses an OpenAI Chat Completions `usage` object — anything
        exposing `.prompt_tokens`/`.completion_tokens`, like
        `openai.types.CompletionUsage` — into `ActualUsage`, so callers
        never need to know this SDK's exact attribute names themselves.
        Cost is always priced from `prompt_tokens`/`completion_tokens`
        exactly as OpenAI reports and bills them — unchanged by, and
        independent of, `parse_openai_usage_detail`'s richer breakdown.
        """
        return self.calculate_actual_usage(
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens
        )

    def parse_openai_usage_detail(self, usage: object) -> TokenUsageDetail:
        """Captures every token-count field OpenAI's `usage` object can
        report — including `completion_tokens_details.reasoning_tokens`
        and `prompt_tokens_details.cached_tokens`, which
        `calculate_actual_usage_from_openai` never surfaces since it
        only prices the top-level totals. Reads every field
        defensively (`getattr(..., None)`): a fake/test usage object,
        or a real one from a model that never returns these detail
        sub-objects, simply yields `None` for them rather than raising.
        """
        completion_details = getattr(usage, "completion_tokens_details", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        return TokenUsageDetail(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", None),
            cached_tokens=getattr(prompt_details, "cached_tokens", None),
            accepted_prediction_tokens=getattr(completion_details, "accepted_prediction_tokens", None),
            rejected_prediction_tokens=getattr(completion_details, "rejected_prediction_tokens", None),
            audio_tokens_prompt=getattr(prompt_details, "audio_tokens", None),
            audio_tokens_completion=getattr(completion_details, "audio_tokens", None),
        )

    def count_tokens(self, text: str) -> int:
        """Counts `text`'s tokens locally via `tiktoken`, using this
        calculator's own chat `model`'s encoding — for diagnostics only
        (e.g. comparing a response's visible answer against OpenAI's
        actual billed `completion_tokens`), never for billing or cost
        math, which always uses OpenAI's own reported usage and must
        never be replaced by a local estimate.
        """
        return len(self._get_encoding().encode(text))

    def _get_encoding(self) -> tiktoken.Encoding:
        if self._encoding is None:
            try:
                self._encoding = tiktoken.encoding_for_model(self._model)
            except KeyError:
                self._encoding = tiktoken.get_encoding(_FALLBACK_ENCODING)
        return self._encoding

    def convert_to_money(self, usd_amount: float) -> MoneyAmount:
        """Converts an already-computed USD amount (e.g. the Upload
        Embedding pipeline's per-1k-token cost, which is priced
        independently of this calculator's per-million-token LLM
        pricing) into the same `{usd, inr}` shape everything else
        uses — so this remains the only place currency conversion
        happens, even for costs this class didn't itself price.
        """
        return self._to_money(usd_amount)

    def pricing_snapshot(self) -> PricingSnapshot:
        return PricingSnapshot(
            model=self._model,
            input_price_per_million_tokens=self._input_price_per_million_tokens,
            output_price_per_million_tokens=self._output_price_per_million_tokens,
            usd_to_inr_exchange_rate=self._usd_to_inr_exchange_rate,
        )

    def _to_money(self, usd_amount: float) -> MoneyAmount:
        return MoneyAmount(usd=usd_amount, inr=round(usd_amount * self._usd_to_inr_exchange_rate, _INR_DECIMAL_PLACES))

    @staticmethod
    def _token_cost(tokens: int, price_per_million_tokens: float) -> float:
        return (tokens / 1_000_000) * price_per_million_tokens
