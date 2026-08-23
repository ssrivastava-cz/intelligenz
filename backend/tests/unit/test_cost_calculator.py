"""Unit tests for CostCalculator — the single place every LLM cost
calculation happens, with pricing and the USD->INR exchange rate always
supplied at construction (never hardcoded) so estimating a prompt's
cost and calculating a completed generation's actual cost use identical
logic. USD is the source of truth; INR is a convenience conversion
derived from it, computed only here.
"""
from types import SimpleNamespace

from app.services.cost_calculator import CostCalculator


def _calculator(
    model: str = "gpt-5",
    input_price: float = 1.25,
    output_price: float = 10.00,
    usd_to_inr_exchange_rate: float = 87.00,
    cached_input_price: float | None = 0.125,
) -> CostCalculator:
    return CostCalculator(
        model=model,
        input_price_per_million_tokens=input_price,
        output_price_per_million_tokens=output_price,
        usd_to_inr_exchange_rate=usd_to_inr_exchange_rate,
        cached_input_price_per_million_tokens=cached_input_price,
    )


def test_model_property_reflects_configured_model():
    calculator = _calculator(model="gpt-5")

    assert calculator.model == "gpt-5"


def test_estimate_input_usage_computes_usd_cost_from_configured_pricing():
    calculator = _calculator(input_price=2.0)

    usage = calculator.estimate_input_usage(input_tokens=1_000_000)

    assert usage.model == "gpt-5"
    assert usage.input_tokens == 1_000_000
    assert usage.estimated_input_cost.usd == 2.0
    assert usage.currency == "USD"


def test_estimate_input_usage_converts_to_inr_using_configured_exchange_rate():
    calculator = _calculator(input_price=1.0, usd_to_inr_exchange_rate=87.0)

    usage = calculator.estimate_input_usage(input_tokens=1_000_000)

    assert usage.estimated_input_cost.usd == 1.0
    assert usage.estimated_input_cost.inr == 87.0


def test_estimate_input_usage_rounds_inr_to_two_decimal_places():
    calculator = _calculator(input_price=1.25, usd_to_inr_exchange_rate=87.0)

    # 23491 tokens at $1.25/M = 0.02936375 USD -> * 87 = 2.55464625 INR
    usage = calculator.estimate_input_usage(input_tokens=23_491)

    assert usage.estimated_input_cost.usd == 0.02936375
    assert usage.estimated_input_cost.inr == 2.55


def test_estimate_input_usage_scales_linearly_with_tokens():
    calculator = _calculator(input_price=1.25)

    usage = calculator.estimate_input_usage(input_tokens=500_000)

    assert usage.estimated_input_cost.usd == 0.625


def test_estimate_input_usage_is_zero_for_zero_tokens():
    calculator = _calculator()

    usage = calculator.estimate_input_usage(input_tokens=0)

    assert usage.estimated_input_cost.usd == 0.0
    assert usage.estimated_input_cost.inr == 0.0


def test_different_exchange_rates_produce_different_inr_for_the_same_usd_cost():
    """Confirms the exchange rate is never a hardcoded module constant —
    two calculators with different configured rates must produce
    different INR amounts for the identical USD cost."""
    low_rate = _calculator(input_price=1.0, usd_to_inr_exchange_rate=80.0)
    high_rate = _calculator(input_price=1.0, usd_to_inr_exchange_rate=90.0)

    assert low_rate.estimate_input_usage(1_000_000).estimated_input_cost.inr == 80.0
    assert high_rate.estimate_input_usage(1_000_000).estimated_input_cost.inr == 90.0


def test_calculate_actual_usage_computes_usd_input_output_and_total_cost():
    calculator = _calculator(input_price=1.25, output_price=10.00)

    usage = calculator.calculate_actual_usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert usage.model == "gpt-5"
    assert usage.prompt_tokens == 1_000_000
    assert usage.completion_tokens == 1_000_000
    assert usage.total_tokens == 2_000_000
    assert usage.input_cost.usd == 1.25
    assert usage.output_cost.usd == 10.00
    assert usage.total_cost.usd == 11.25


def test_calculate_actual_usage_computes_inr_for_every_cost_field():
    calculator = _calculator(input_price=1.25, output_price=10.00, usd_to_inr_exchange_rate=87.0)

    usage = calculator.calculate_actual_usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert usage.input_cost.inr == 108.75  # 1.25 * 87
    assert usage.output_cost.inr == 870.0  # 10.00 * 87
    assert usage.total_cost.inr == 978.75  # sum of the two rounded parts


def test_calculate_actual_usage_total_inr_is_sum_of_rounded_parts_not_a_fresh_conversion():
    """Matches the worked example from the spec: promptTokens=23501,
    completionTokens=4123 at $1.25/$10 per million and an 87 rate yields
    inputCost=$0.02938/INR2.56, outputCost=$0.04123/INR3.59, and a total
    of INR6.15 — the sum of the two rounded INR parts, not
    round(0.07061 * 87, 2), which would be 6.14.
    """
    calculator = _calculator(input_price=1.25, output_price=10.00, usd_to_inr_exchange_rate=87.0)

    usage = calculator.calculate_actual_usage(prompt_tokens=23_501, completion_tokens=4_123)

    assert usage.input_cost.usd == 0.02937625
    assert usage.output_cost.usd == 0.04123
    assert usage.input_cost.inr == 2.56
    assert usage.output_cost.inr == 3.59
    assert usage.total_cost.usd == 0.02937625 + 0.04123
    assert usage.total_cost.inr == 6.15


def test_calculate_actual_usage_total_tokens_is_never_trusted_from_the_caller():
    """total_tokens is always derived from prompt + completion tokens,
    not accepted as a separate (possibly inconsistent) input."""
    calculator = _calculator()

    usage = calculator.calculate_actual_usage(prompt_tokens=100, completion_tokens=50)

    assert usage.total_tokens == 150


def test_calculate_actual_usage_from_openai_parses_prompt_and_completion_tokens():
    calculator = _calculator(input_price=1.25, output_price=10.00)
    fake_openai_usage = SimpleNamespace(prompt_tokens=2_000_000, completion_tokens=100_000, total_tokens=2_100_000)

    usage = calculator.calculate_actual_usage_from_openai(fake_openai_usage)

    assert usage.prompt_tokens == 2_000_000
    assert usage.completion_tokens == 100_000
    assert usage.total_tokens == 2_100_000
    assert usage.input_cost.usd == 2.5
    assert usage.output_cost.usd == 1.0
    assert usage.total_cost.usd == 3.5


def test_calculate_actual_usage_from_openai_ignores_completion_tokens_details_for_cost():
    """Reasoning tokens are already included in `completion_tokens` by
    OpenAI's own accounting (billed as output, not separately) — cost
    must come from the top-level totals alone, never double-counting or
    re-deriving from `completion_tokens_details`.
    """
    calculator = _calculator(input_price=1.25, output_price=10.00)
    fake_openai_usage = SimpleNamespace(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=900_000, accepted_prediction_tokens=0),
    )

    usage = calculator.calculate_actual_usage_from_openai(fake_openai_usage)

    assert usage.output_cost.usd == 10.00


# --- parse_openai_usage_detail: the reported bug (gen_20260813_9c771 —
# 1,094 output_tokens for a ~60-100 word visible answer) ---


def test_parse_openai_usage_detail_captures_reasoning_and_cached_tokens():
    calculator = _calculator()
    fake_openai_usage = SimpleNamespace(
        prompt_tokens=1568,
        completion_tokens=1094,
        total_tokens=2662,
        completion_tokens_details=SimpleNamespace(
            reasoning_tokens=960,
            accepted_prediction_tokens=0,
            rejected_prediction_tokens=0,
            audio_tokens=0,
        ),
        prompt_tokens_details=SimpleNamespace(cached_tokens=128, audio_tokens=0),
    )

    detail = calculator.parse_openai_usage_detail(fake_openai_usage)

    assert detail.prompt_tokens == 1568
    assert detail.completion_tokens == 1094
    assert detail.total_tokens == 2662
    assert detail.reasoning_tokens == 960
    assert detail.cached_tokens == 128
    assert detail.accepted_prediction_tokens == 0
    assert detail.rejected_prediction_tokens == 0
    assert detail.audio_tokens_prompt == 0
    assert detail.audio_tokens_completion == 0


def test_parse_openai_usage_detail_leaves_detail_fields_none_when_the_api_omits_them():
    """Not every model/response includes `completion_tokens_details`/
    `prompt_tokens_details` — absence must mean `None`, never `0` (which
    would falsely claim "confirmed zero reasoning tokens")."""
    calculator = _calculator()
    fake_openai_usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    detail = calculator.parse_openai_usage_detail(fake_openai_usage)

    assert detail.prompt_tokens == 100
    assert detail.completion_tokens == 50
    assert detail.total_tokens == 150
    assert detail.reasoning_tokens is None
    assert detail.cached_tokens is None
    assert detail.accepted_prediction_tokens is None
    assert detail.rejected_prediction_tokens is None


def test_parse_openai_usage_detail_never_alters_the_top_level_totals():
    """The raw API totals are preserved exactly, regardless of whether
    a detailed breakdown exists alongside them."""
    calculator = _calculator()
    with_detail = SimpleNamespace(
        prompt_tokens=200,
        completion_tokens=300,
        total_tokens=500,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=250),
    )
    without_detail = SimpleNamespace(prompt_tokens=200, completion_tokens=300, total_tokens=500)

    detail_with = calculator.parse_openai_usage_detail(with_detail)
    detail_without = calculator.parse_openai_usage_detail(without_detail)

    assert detail_with.completion_tokens == detail_without.completion_tokens == 300


# --- count_tokens: local diagnostic-only token counting, never used for cost ---


def test_count_tokens_returns_a_positive_count_for_non_empty_text():
    calculator = _calculator(model="gpt-5")

    assert calculator.count_tokens("Users with delete permissions can remove a contact log entry.") > 0


def test_count_tokens_returns_zero_for_empty_text():
    calculator = _calculator()

    assert calculator.count_tokens("") == 0


def test_count_tokens_is_deterministic_for_the_same_text():
    calculator = _calculator()
    text = "A short visible answer, about a dozen words long in total."

    assert calculator.count_tokens(text) == calculator.count_tokens(text)


def test_count_tokens_grows_with_longer_text():
    calculator = _calculator()
    short_text = "Short answer."
    long_text = " ".join(["This is a much longer answer with many more words in it."] * 20)

    assert calculator.count_tokens(long_text) > calculator.count_tokens(short_text)


def test_count_tokens_falls_back_to_a_known_encoding_for_an_unrecognized_model():
    calculator = _calculator(model="not-a-real-openai-model-name")

    # Must not raise — falls back to a default encoding rather than
    # propagating tiktoken's KeyError for an unknown model name.
    assert calculator.count_tokens("Some text.") > 0


def test_pricing_snapshot_reflects_configured_model_prices_and_exchange_rate():
    calculator = _calculator(model="gpt-5", input_price=1.25, output_price=10.00, usd_to_inr_exchange_rate=87.0)

    snapshot = calculator.pricing_snapshot()

    assert snapshot.model == "gpt-5"
    assert snapshot.input_price_per_million_tokens == 1.25
    assert snapshot.output_price_per_million_tokens == 10.00
    assert snapshot.usd_to_inr_exchange_rate == 87.0


def test_different_calculator_instances_use_independent_pricing():
    """Confirms pricing is never a hardcoded module-level constant —
    two calculators with different configured prices must produce
    different costs for the same token count."""
    cheap = _calculator(input_price=1.0)
    expensive = _calculator(input_price=100.0)

    assert cheap.estimate_input_usage(1_000_000).estimated_input_cost.usd == 1.0
    assert expensive.estimate_input_usage(1_000_000).estimated_input_cost.usd == 100.0
