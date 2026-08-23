"""Unit tests for GenerationSummary.from_entry — the single place that
flattens a full GenerationHistoryEntry into the trimmed fields the
`/generation/history` listing and `HistoryService.generation_statistics`
both need. The Embedding-stage join (matching `metadata.upload_session_id`
against Upload Embedding history) is `HistoryService`'s job, not
`from_entry`'s — these tests exercise `from_entry` with already-resolved
`embedding_tokens`/`embedding_cost` arguments, exactly as
`HistoryService.list_generation_summaries` supplies them.
"""
from datetime import UTC, datetime

from app.models.cost import ActualUsage, EstimatedUsage, MoneyAmount, PricingSnapshot
from app.models.generated_test_case import GeneratedTestCase, GeneratedTestStep
from app.models.generation_history import GenerationHistoryEntry, GenerationMetadata, RetrievalSummary
from app.models.generation_summary import GenerationSummary


def _make_test_case(**overrides) -> GeneratedTestCase:
    base = {
        "requirement_id": "REQ-1",
        "test_case_id": "TC-1",
        "test_case_title": "Reschedule an appointment",
        "priority": "High",
        "test_suite": "Appointments",
        "preconditions": "User is logged in.",
        "steps": [GeneratedTestStep(step_no=1, action="Open appointment.", expected_result="Details are shown.")],
        "post_conditions": "Appointment is updated.",
        "automation_status": "Not Automated",
        "test_type": "Functional",
        "tags": ["Appointments"],
    }
    base.update(overrides)
    return GeneratedTestCase(**base)


def _make_entry(**overrides) -> GenerationHistoryEntry:
    base = {
        "generation_id": "2026-08-03T21-15-42",
        "timestamp": datetime(2026, 8, 3, 21, 15, 42, tzinfo=UTC),
        "feature": "Appointments",
        "redmine_ticket": "12345",
        "model": "gpt-5",
        "prompt_version": "1.0.0",
        "retrieval": RetrievalSummary(
            workflow_chunks=1, historical_test_cases=1, historical_issues=0, uploaded_documents=0
        ),
        "estimated_usage": EstimatedUsage(
            model="gpt-5", input_tokens=1000, estimated_input_cost=MoneyAmount(usd=0.00125, inr=0.11)
        ),
        "actual_usage": ActualUsage(
            model="gpt-5",
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
            input_cost=MoneyAmount(usd=0.00125, inr=0.11),
            output_cost=MoneyAmount(usd=0.002, inr=0.17),
            total_cost=MoneyAmount(usd=0.00325, inr=0.28),
        ),
        "pricing": PricingSnapshot(
            model="gpt-5",
            input_price_per_million_tokens=1.25,
            output_price_per_million_tokens=10.00,
            usd_to_inr_exchange_rate=87.00,
        ),
        "generation_time_ms": 2300.0,
        "prompt": "Prompt Version: 1.0.0\n\n...",
        "response": '{"testCases": []}',
        "test_cases": [_make_test_case(), _make_test_case(test_case_id="TC-2")],
        "metadata": GenerationMetadata(top_k=5, upload_session_id="sess-abc123", generated_test_cases=2),
    }
    base.update(overrides)
    return GenerationHistoryEntry(**base)


def test_from_entry_maps_identifying_and_scalar_fields():
    summary = GenerationSummary.from_entry(_make_entry())

    assert summary.generation_id == "2026-08-03T21-15-42"
    assert summary.created_at == datetime(2026, 8, 3, 21, 15, 42, tzinfo=UTC)
    assert summary.feature == "Appointments"
    assert summary.redmine_ticket == "12345"
    assert summary.model == "gpt-5"
    assert summary.prompt_version == "1.0.0"
    assert summary.generation_time_ms == 2300.0


def test_from_entry_counts_test_cases_from_the_actual_list_not_metadata():
    entry = _make_entry(metadata=GenerationMetadata(top_k=5, upload_session_id=None, generated_test_cases=999))

    summary = GenerationSummary.from_entry(entry)

    assert summary.number_of_test_cases == 2


def test_from_entry_flattens_nested_actual_usage_fields():
    summary = GenerationSummary.from_entry(_make_entry())

    assert summary.prompt_tokens == 1000
    assert summary.completion_tokens == 200
    assert summary.total_tokens == 1200


def test_from_entry_flattens_actual_generation_cost_fields():
    summary = GenerationSummary.from_entry(_make_entry())

    assert summary.actual_generation_cost_usd == 0.00325
    assert summary.actual_generation_cost_inr == 0.28


def test_from_entry_flattens_estimated_prompt_fields():
    summary = GenerationSummary.from_entry(_make_entry())

    assert summary.estimated_prompt_tokens == 1000
    assert summary.estimated_input_cost_usd == 0.00125
    assert summary.estimated_input_cost_inr == 0.11


def test_from_entry_defaults_embedding_fields_to_none_when_not_supplied():
    summary = GenerationSummary.from_entry(_make_entry())

    assert summary.embedding_tokens is None
    assert summary.embedding_cost_usd is None
    assert summary.embedding_cost_inr is None


def test_from_entry_flattens_supplied_embedding_fields():
    summary = GenerationSummary.from_entry(
        _make_entry(),
        embedding_tokens=500,
        embedding_cost=MoneyAmount(usd=0.00001, inr=0.0009),
    )

    assert summary.embedding_tokens == 500
    assert summary.embedding_cost_usd == 0.00001
    assert summary.embedding_cost_inr == 0.0009


def test_from_entry_total_ai_cost_is_actual_generation_cost_when_no_embedding():
    summary = GenerationSummary.from_entry(_make_entry())

    assert summary.total_ai_cost_usd == summary.actual_generation_cost_usd
    assert summary.total_ai_cost_inr == summary.actual_generation_cost_inr


def test_from_entry_total_ai_cost_is_embedding_plus_actual_generation_cost():
    summary = GenerationSummary.from_entry(
        _make_entry(),
        embedding_tokens=500,
        embedding_cost=MoneyAmount(usd=0.00001, inr=0.0009),
    )

    assert summary.total_ai_cost_usd == 0.00001 + 0.00325
    assert summary.total_ai_cost_inr == 0.0009 + 0.28


def test_from_entry_defaults_status_to_success():
    summary = GenerationSummary.from_entry(_make_entry())

    assert summary.status == "SUCCESS"


def test_from_entry_handles_zero_test_cases():
    entry = _make_entry(
        test_cases=[], metadata=GenerationMetadata(top_k=None, upload_session_id=None, generated_test_cases=0)
    )

    summary = GenerationSummary.from_entry(entry)

    assert summary.number_of_test_cases == 0
