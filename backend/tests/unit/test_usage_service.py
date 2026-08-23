"""Unit tests for `UsageService` — normalizes and combines Test Plan
Generator history (`HistoryService`) and Knowledge Assistant usage
(`GenerationRepository`) into one unified Usage Dashboard view. Uses
the real `HistoryService`/`FileSystemGenerationRepository` rooted at
`tmp_path`, exactly like `tests/unit/test_history_service.py` and
`tests/unit/test_filesystem_generation_repository.py`, so these tests
exercise the real normalization path rather than a hand-rolled fake.
"""
from app.models.cost import ActualUsage, EstimatedUsage, MoneyAmount, PricingSnapshot
from app.models.generated_test_case import GeneratedTestCase, GeneratedTestStep
from app.models.generation_history import GenerationHistoryDraft, GenerationMetadata, RetrievalSummary
from app.models.normalized_usage import UsageSource
from app.repositories.filesystem_generation_repository import FileSystemGenerationRepository
from app.services.cost_calculator import CostCalculator
from app.services.history_service import HistoryService
from app.services.usage_service import UsageService


def _cost_calculator() -> CostCalculator:
    return CostCalculator(
        model="gpt-5",
        input_price_per_million_tokens=1.25,
        output_price_per_million_tokens=10.00,
        usd_to_inr_exchange_rate=87.00,
    )


def _make_generated_test_case(**overrides) -> GeneratedTestCase:
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


def _make_generation_draft(
    feature: str = "Appointments", total_cost_usd: float = 1.0, **overrides
) -> GenerationHistoryDraft:
    base = {
        "feature": feature,
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
            input_cost=MoneyAmount(usd=total_cost_usd * 0.5, inr=total_cost_usd * 0.5 * 87),
            output_cost=MoneyAmount(usd=total_cost_usd * 0.5, inr=total_cost_usd * 0.5 * 87),
            total_cost=MoneyAmount(usd=total_cost_usd, inr=total_cost_usd * 87),
        ),
        "pricing": PricingSnapshot(
            model="gpt-5",
            input_price_per_million_tokens=1.25,
            output_price_per_million_tokens=10.00,
            usd_to_inr_exchange_rate=87.00,
        ),
        "generation_time_ms": 1000.0,
        "prompt": "Prompt Version: 1.0.0\n\n...",
        "response": '{"testCases": [{"testCaseId": "TC-1"}]}',
        "test_cases": [_make_generated_test_case()],
        "metadata": GenerationMetadata(top_k=5, upload_session_id=None, generated_test_cases=1),
    }
    base.update(overrides)
    return GenerationHistoryDraft(**base)


def _service(tmp_path) -> tuple[UsageService, HistoryService, FileSystemGenerationRepository]:
    history_service = HistoryService(history_root=tmp_path / "history")
    generation_repository = FileSystemGenerationRepository(database_root=tmp_path / "database")
    service = UsageService(
        history_service=history_service,
        generation_repository=generation_repository,
        cost_calculator=_cost_calculator(),
    )
    return service, history_service, generation_repository


def test_get_test_plan_generation_usage_normalizes_summary_fields(tmp_path):
    service, history_service, _ = _service(tmp_path)
    saved = history_service.save_generation_history(_make_generation_draft(total_cost_usd=1.0))

    [record] = service.get_test_plan_generation_usage()

    assert record.generation_id == saved.generation_id
    assert record.source == UsageSource.TEST_PLAN_GENERATOR
    assert record.model == "gpt-5"
    assert record.input_tokens == 1000
    assert record.output_tokens == 200
    assert record.total_tokens == 1200
    assert record.estimated_input_tokens == 1000
    # AI Generation stage only: half of the 87.0 INR total each.
    assert round(record.input_cost_inr, 2) == 43.5
    assert round(record.output_cost_inr, 2) == 43.5
    # total_cost_inr uses total_ai_cost_inr (embedding + generation) —
    # with no associated upload session, that's the generation cost alone.
    assert round(record.total_cost_inr, 2) == 87.0
    assert record.generation_time_ms == 1000.0


def test_get_knowledge_assistant_usage_normalizes_usage_record_fields(tmp_path):
    service, _, generation_repository = _service(tmp_path)
    generation_repository.save_usage(
        "gen_ka_1",
        model="gpt-5",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        estimated_input_tokens=120,
        input_cost_inr=0.10,
        output_cost_inr=0.20,
        total_cost_inr=0.30,
        generation_time_ms=900.0,
    )

    [record] = service.get_knowledge_assistant_usage()

    assert record.generation_id == "gen_ka_1"
    assert record.source == UsageSource.KNOWLEDGE_ASSISTANT
    assert record.input_tokens == 100
    assert record.output_tokens == 50
    assert record.total_tokens == 150
    assert record.estimated_input_tokens == 120
    assert record.input_cost_inr == 0.10
    assert record.output_cost_inr == 0.20
    assert record.total_cost_inr == 0.30
    assert record.generation_time_ms == 900.0


def test_get_knowledge_assistant_usage_normalizes_the_detailed_token_breakdown(tmp_path):
    service, _, generation_repository = _service(tmp_path)
    generation_repository.save_usage(
        "gen_ka_1",
        model="gpt-5",
        input_tokens=1568,
        output_tokens=1094,
        total_tokens=2662,
        estimated_input_tokens=1431,
        input_cost_inr=0.19,
        output_cost_inr=1.04,
        total_cost_inr=1.23,
        generation_time_ms=22274.7,
        reasoning_tokens=960,
        cached_tokens=128,
        visible_answer_tokens=78,
        structured_output_overhead_tokens=56,
    )

    [record] = service.get_knowledge_assistant_usage()

    assert record.reasoning_tokens == 960
    assert record.cached_tokens == 128
    assert record.visible_answer_tokens == 78
    assert record.structured_output_overhead_tokens == 56


def test_get_knowledge_assistant_usage_leaves_the_detailed_breakdown_none_when_not_tracked(tmp_path):
    service, _, generation_repository = _service(tmp_path)
    generation_repository.save_usage("gen_ka_1", "gpt-5", 100, 50, 150, 120, 0.1, 0.2, 0.3, 900.0)

    [record] = service.get_knowledge_assistant_usage()

    assert record.reasoning_tokens is None
    assert record.cached_tokens is None
    assert record.visible_answer_tokens is None
    assert record.structured_output_overhead_tokens is None


def test_get_test_plan_generation_usage_normalizes_with_no_detailed_token_breakdown(tmp_path):
    """Test Plan Generator rows never carry the detailed breakdown —
    `None` (not tracked for that source), never a fabricated `0`."""
    service, history_service, _ = _service(tmp_path)
    history_service.save_generation_history(_make_generation_draft())

    [record] = service.get_test_plan_generation_usage()

    assert record.reasoning_tokens is None
    assert record.cached_tokens is None
    assert record.visible_answer_tokens is None
    assert record.structured_output_overhead_tokens is None


def test_get_test_plan_generation_usage_returns_empty_list_when_no_history_exists(tmp_path):
    service, _, _ = _service(tmp_path)

    assert service.get_test_plan_generation_usage() == []


def test_get_knowledge_assistant_usage_returns_empty_list_when_no_generations_exist(tmp_path):
    service, _, _ = _service(tmp_path)

    assert service.get_knowledge_assistant_usage() == []


def test_get_usage_returns_zeroed_summary_when_both_sources_are_empty(tmp_path):
    service, _, _ = _service(tmp_path)

    result = service.get_usage()

    assert result.summary.total_generations == 0
    assert result.summary.total_tokens == 0
    assert result.summary.total_ai_cost_inr == 0.0
    assert result.summary.average_cost_inr == 0.0
    assert result.generations == []


def test_get_usage_combines_and_sorts_both_sources_newest_first(tmp_path):
    service, history_service, generation_repository = _service(tmp_path)
    history_service.save_generation_history(_make_generation_draft(feature="Older"))
    generation_repository.save_usage(
        "gen_ka_newer", "gpt-5", 100, 50, 150, 120, 0.1, 0.2, 0.3, 900.0
    )

    result = service.get_usage()

    created_ats = [record.created_at for record in result.generations]
    assert created_ats == sorted(created_ats, reverse=True)
    assert {record.source for record in result.generations} == {
        UsageSource.TEST_PLAN_GENERATOR,
        UsageSource.KNOWLEDGE_ASSISTANT,
    }


def test_get_usage_summary_sums_generations_tokens_and_cost_across_both_sources(tmp_path):
    service, history_service, generation_repository = _service(tmp_path)
    history_service.save_generation_history(_make_generation_draft(total_cost_usd=1.0))  # 1200 tokens, 87.0 INR
    generation_repository.save_usage("gen_ka_1", "gpt-5", 100, 50, 150, 120, 0.1, 0.2, 13.0, 900.0)

    result = service.get_usage()

    assert result.summary.total_generations == 2
    assert result.summary.total_tokens == 1200 + 150
    assert round(result.summary.total_ai_cost_inr, 2) == round(87.0 + 13.0, 2)
    assert round(result.summary.average_cost_inr, 2) == round((87.0 + 13.0) / 2, 2)


def test_get_test_plan_generation_usage_does_not_break_when_knowledge_assistant_source_is_unavailable(
    tmp_path, monkeypatch
):
    """If the Knowledge Assistant side fails entirely, the Test Plan
    Generator side must still be usable — and vice versa."""
    service, history_service, generation_repository = _service(tmp_path)
    history_service.save_generation_history(_make_generation_draft())

    def _raise():
        raise OSError("Knowledge Assistant storage unavailable.")

    monkeypatch.setattr(generation_repository, "list_usage_records", _raise)

    result = service.get_usage()

    assert result.summary.total_generations == 1
    assert result.generations[0].source == UsageSource.TEST_PLAN_GENERATOR


def test_get_knowledge_assistant_usage_does_not_break_when_test_plan_generator_source_is_unavailable(
    tmp_path, monkeypatch
):
    service, history_service, generation_repository = _service(tmp_path)
    generation_repository.save_usage("gen_ka_1", "gpt-5", 100, 50, 150, 120, 0.1, 0.2, 0.3, 900.0)

    def _raise(cost_calculator):
        raise OSError("Test Plan Generator history unavailable.")

    monkeypatch.setattr(history_service, "list_generation_summaries", _raise)

    result = service.get_usage()

    assert result.summary.total_generations == 1
    assert result.generations[0].source == UsageSource.KNOWLEDGE_ASSISTANT
