"""Unit tests for `HistoryService` — the only component allowed to write
into `backend/data/history/`. Covers indexing history (`history/index/`),
uploaded-document embedding history (`history/upload/`), and AI
generation history (`history/generation/`).
"""
import json
import re

import pytest

from app.core.exceptions import ExternalServiceError, NotFoundError
from app.models.cost import ActualUsage, EstimatedUsage, MoneyAmount, PricingSnapshot
from app.models.embedding_preview import EmbeddingPreviewChunk
from app.models.generated_test_case import GeneratedTestCase, GeneratedTestStep
from app.models.generation_history import GenerationHistoryDraft, GenerationMetadata, RetrievalSummary
from app.models.index_history import IndexHistoryEntry, IndexHistoryRecord
from app.models.upload_history import UploadHistoryEntry
from app.services.cost_calculator import CostCalculator
from app.services.history_service import HistoryService
from app.utils.datetime_utils import utcnow

_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(_\d+)?")
_GENERATION_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(_\d+)?")


def _cost_calculator() -> CostCalculator:
    return CostCalculator(
        model="gpt-5",
        input_price_per_million_tokens=1.25,
        output_price_per_million_tokens=10.00,
        usd_to_inr_exchange_rate=87.00,
    )


def _make_index_entry(feature: str = "Appointments", **overrides) -> IndexHistoryEntry:
    base = {
        "feature": feature,
        "indexed_at": utcnow(),
        "embedding_model": "text-embedding-3-small",
        "documents_indexed": 1,
        "chunks_indexed": 1,
        "embedding_tokens": 10,
        "average_tokens_per_chunk": 10.0,
        "estimated_embedding_cost": 0.0002,
        "elapsed_seconds": 0.5,
        "chroma_collection_name": "source_of_truth_chunks",
    }
    base.update(overrides)
    return IndexHistoryEntry(**base)


def _make_upload_entry(upload_session_id: str = "sess-abc123", **overrides) -> UploadHistoryEntry:
    base = {
        "upload_session_id": upload_session_id,
        "uploaded_at": utcnow(),
        "embedding_model": "text-embedding-3-small",
        "documents_indexed": 1,
        "chunks_indexed": 1,
        "embedding_tokens": 10,
        "average_tokens_per_chunk": 10.0,
        "estimated_embedding_cost": 0.0002,
        "elapsed_seconds": 0.5,
        "chroma_collection_name": f"uploaded_documents_{upload_session_id}",
    }
    base.update(overrides)
    return UploadHistoryEntry(**base)


def _make_generated_test_case(**overrides) -> GeneratedTestCase:
    base = {
        "requirement_id": "REQ-1",
        "test_case_id": "TC-1",
        "test_case_title": "Reschedule an appointment",
        "priority": "High",
        "test_suite": "Appointments",
        "preconditions": "User is logged in.",
        "steps": [GeneratedTestStep(step_no=1, action="Open appointment.", expected_result="Details shown.")],
        "post_conditions": "Appointment updated.",
        "automation_status": "Not Automated",
        "test_type": "Functional",
        "tags": ["Appointments"],
    }
    base.update(overrides)
    return GeneratedTestCase(**base)


def _make_generation_draft(feature: str = "Appointments", **overrides) -> GenerationHistoryDraft:
    base = {
        "feature": feature,
        "redmine_ticket": "12345",
        "model": "gpt-5",
        "prompt_version": "1.0.0",
        "retrieval": RetrievalSummary(
            workflow_chunks=1, historical_test_cases=1, historical_issues=1, uploaded_documents=0
        ),
        "estimated_usage": EstimatedUsage(
            model="gpt-5",
            input_tokens=1000,
            estimated_input_cost=MoneyAmount(usd=0.00125, inr=0.11),
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
        "response": '{"testCases": [{"testCaseId": "TC-1"}]}',
        "test_cases": [_make_generated_test_case()],
        "metadata": GenerationMetadata(top_k=5, upload_session_id=None, generated_test_cases=1),
    }
    base.update(overrides)
    return GenerationHistoryDraft(**base)


def _make_preview_chunk(**overrides) -> EmbeddingPreviewChunk:
    base = {
        "chunk_id": "chunk-1",
        "section_heading": "Role Permissions",
        "artifact_type": "WORKFLOW",
        "source_filename": "onboarding.md",
        "page_number": None,
        "word_count": 10,
        "embedding_tokens": 12,
        "estimated_cost": 0.0000002,
        "chunk_text": "Role Permissions\n\nBody.",
    }
    base.update(overrides)
    return EmbeddingPreviewChunk(**base)


def test_save_index_history_creates_a_timestamped_folder_under_index_subfolder(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    folder = service.save_index_history(_make_index_entry(), [_make_preview_chunk()])

    assert folder.parent == tmp_path / "history" / "index"
    assert _TIMESTAMP_PATTERN.fullmatch(folder.name)
    assert folder.is_dir()


def test_save_index_history_does_not_create_retrieval_or_generation_folders(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)

    service.save_index_history(_make_index_entry(), [])

    assert not (history_root / "retrieval").exists()
    assert not (history_root / "generation").exists()


def test_save_index_history_writes_index_summary_json_with_all_required_fields(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    entry = _make_index_entry(feature="Appointments", chunks_indexed=5)

    folder = service.save_index_history(entry, [_make_preview_chunk()])

    summary = json.loads((folder / "index_summary.json").read_text(encoding="utf-8"))
    assert summary["feature"] == "Appointments"
    assert summary["indexed_at"]
    assert summary["embedding_model"] == "text-embedding-3-small"
    assert summary["documents_indexed"] == 1
    assert summary["chunks_indexed"] == 5
    assert summary["embedding_tokens"] == 10
    assert summary["average_tokens_per_chunk"] == entry.average_tokens_per_chunk
    assert summary["estimated_embedding_cost"] == entry.estimated_embedding_cost
    assert summary["elapsed_seconds"] == entry.elapsed_seconds
    assert summary["chroma_collection_name"] == "source_of_truth_chunks"
    assert summary["index_status"] == "SUCCESS"


def test_save_index_history_writes_embedding_preview_with_all_required_fields(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    preview = [_make_preview_chunk(chunk_id="chunk-1", page_number=3)]

    folder = service.save_index_history(_make_index_entry(), preview)

    [chunk] = json.loads((folder / "embedding_preview.json").read_text(encoding="utf-8"))
    assert chunk["chunk_id"] == "chunk-1"
    assert chunk["section_heading"] == "Role Permissions"
    assert chunk["artifact_type"] == "WORKFLOW"
    assert chunk["source_filename"] == "onboarding.md"
    assert chunk["page_number"] == 3
    assert chunk["word_count"] == 10
    assert chunk["embedding_tokens"] == 12
    assert chunk["estimated_cost"] == pytest.approx(0.0000002)
    assert chunk["chunk_text"] == "Role Permissions\n\nBody."


def test_save_index_history_never_stores_embedding_vectors(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    preview = [_make_preview_chunk(chunk_id="chunk-1"), _make_preview_chunk(chunk_id="chunk-2")]

    folder = service.save_index_history(_make_index_entry(), preview)

    raw = json.loads((folder / "embedding_preview.json").read_text(encoding="utf-8"))
    assert len(raw) == 2
    assert all("embedding" not in chunk for chunk in raw)
    assert all("vector" not in chunk for chunk in raw)


def test_save_index_history_does_not_collide_on_repeated_calls(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    folder_one = service.save_index_history(_make_index_entry(), [])
    folder_two = service.save_index_history(_make_index_entry(), [])

    assert folder_one != folder_two
    assert folder_one.is_dir()
    assert folder_two.is_dir()


def test_list_index_history_returns_every_saved_entry(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    service.save_index_history(_make_index_entry(feature="Appointments"), [])
    service.save_index_history(_make_index_entry(feature="Coding Tool"), [])

    entries = service.list_index_history()

    assert {entry.feature for entry in entries} == {"Appointments", "Coding Tool"}


def test_list_index_history_returns_empty_list_when_nothing_saved(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    assert service.list_index_history() == []


def test_load_index_history_returns_the_full_record(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    entry = _make_index_entry(feature="Appointments")
    preview = [_make_preview_chunk(chunk_id="chunk-1")]
    folder = service.save_index_history(entry, preview)

    record = service.load_index_history(folder.name)

    assert isinstance(record, IndexHistoryRecord)
    assert record.history_id == folder.name
    assert record.summary.feature == "Appointments"
    assert len(record.embedding_preview) == 1
    assert record.embedding_preview[0].chunk_id == "chunk-1"


def test_load_index_history_raises_not_found_for_unknown_id(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    with pytest.raises(NotFoundError):
        service.load_index_history("2000-01-01_00-00-00")


def test_delete_index_history_removes_the_folder(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    folder = service.save_index_history(_make_index_entry(), [])

    service.delete_index_history(folder.name)

    assert not folder.exists()
    assert service.list_index_history() == []


def test_delete_index_history_raises_not_found_for_unknown_id(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    with pytest.raises(NotFoundError):
        service.delete_index_history("2000-01-01_00-00-00")


def test_delete_index_history_only_removes_the_targeted_run(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    keep = service.save_index_history(_make_index_entry(feature="Appointments"), [])
    remove = service.save_index_history(_make_index_entry(feature="Coding Tool"), [])

    service.delete_index_history(remove.name)

    assert keep.exists()
    remaining = service.list_index_history()
    assert [entry.feature for entry in remaining] == ["Appointments"]


def test_list_index_history_skips_a_corrupted_file_without_failing(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)
    service.save_index_history(_make_index_entry(feature="Appointments"), [])
    index_root = history_root / "index"
    (index_root / "2000-01-01_00-00-00").mkdir()
    (index_root / "2000-01-01_00-00-00" / "index_summary.json").write_text("{not valid json", encoding="utf-8")
    (index_root / "2000-01-02_00-00-00").mkdir()
    (index_root / "2000-01-02_00-00-00" / "index_summary.json").write_text(
        json.dumps({"unexpected": "shape"}), encoding="utf-8"
    )

    entries = service.list_index_history()

    assert len(entries) == 1
    assert entries[0].feature == "Appointments"


def test_list_index_history_excludes_non_success_entries(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    service.save_index_history(_make_index_entry(feature="Appointments", index_status="SUCCESS"), [])
    service.save_index_history(_make_index_entry(feature="Failed Run", index_status="FAILED"), [])

    entries = service.list_index_history()

    assert [entry.feature for entry in entries] == ["Appointments"]


# --- list_indexing_summaries / indexing_statistics ---


def test_list_indexing_summaries_converts_embedding_cost_via_the_cost_calculator(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    service.save_index_history(_make_index_entry(feature="Appointments", estimated_embedding_cost=0.00044146), [])

    [summary] = service.list_indexing_summaries(_cost_calculator())

    assert summary.embedding_cost_usd == pytest.approx(0.00044146)
    assert summary.embedding_cost_inr == pytest.approx(round(0.00044146 * 87.00, 2))


def test_list_indexing_summaries_includes_the_history_id_and_activity_type(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    folder = service.save_index_history(_make_index_entry(feature="Appointments"), [])

    [summary] = service.list_indexing_summaries(_cost_calculator())

    assert summary.history_id == folder.name
    assert summary.activity_type == "DOCUMENT_INDEXING"
    assert summary.status == "SUCCESS"


def test_list_indexing_summaries_skips_corrupted_and_excludes_non_success_entries(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)
    service.save_index_history(_make_index_entry(feature="Appointments"), [])
    service.save_index_history(_make_index_entry(feature="Failed Run", index_status="FAILED"), [])
    index_root = history_root / "index"
    (index_root / "2000-01-01_00-00-00").mkdir()
    (index_root / "2000-01-01_00-00-00" / "index_summary.json").write_text("{not valid json", encoding="utf-8")

    summaries = service.list_indexing_summaries(_cost_calculator())

    assert len(summaries) == 1
    assert summaries[0].feature == "Appointments"


def test_list_indexing_summaries_returns_empty_list_when_nothing_saved(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    assert service.list_indexing_summaries(_cost_calculator()) == []


def test_indexing_statistics_returns_zeroed_stats_when_nothing_saved(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    stats = service.indexing_statistics(_cost_calculator())

    assert stats.total_indexing_runs == 0
    assert stats.total_documents_indexed == 0
    assert stats.total_chunks_indexed == 0
    assert stats.total_embedding_tokens == 0
    assert stats.total_embedding_cost_usd == 0.0
    assert stats.total_embedding_cost_inr == 0.0


def test_indexing_statistics_aggregates_totals_across_multiple_runs(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    service.save_index_history(
        _make_index_entry(
            feature="Appointments",
            documents_indexed=6,
            chunks_indexed=161,
            embedding_tokens=22073,
            estimated_embedding_cost=0.00044146,
        ),
        [],
    )
    service.save_index_history(
        _make_index_entry(
            feature="Coding Tool",
            documents_indexed=3,
            chunks_indexed=40,
            embedding_tokens=5000,
            estimated_embedding_cost=0.0001,
        ),
        [],
    )

    stats = service.indexing_statistics(_cost_calculator())

    assert stats.total_indexing_runs == 2
    assert stats.total_documents_indexed == 9
    assert stats.total_chunks_indexed == 201
    assert stats.total_embedding_tokens == 27073
    assert stats.total_embedding_cost_usd == pytest.approx(0.00054146)
    assert stats.total_embedding_cost_inr == pytest.approx(
        round(0.00044146 * 87.00, 2) + round(0.0001 * 87.00, 2)
    )


def test_indexing_statistics_excludes_corrupted_and_failed_runs(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)
    service.save_index_history(_make_index_entry(feature="Appointments", documents_indexed=1), [])
    service.save_index_history(_make_index_entry(feature="Failed Run", index_status="FAILED"), [])
    index_root = history_root / "index"
    (index_root / "2000-01-01_00-00-00").mkdir()
    (index_root / "2000-01-01_00-00-00" / "index_summary.json").write_text("{not valid json", encoding="utf-8")

    stats = service.indexing_statistics(_cost_calculator())

    assert stats.total_indexing_runs == 1
    assert stats.total_documents_indexed == 1


def test_indexing_history_never_affects_generation_statistics_or_total_spend(tmp_path):
    """Indexing and generation are different AI activities with
    different costs — an indexing run's embedding cost must never be
    folded into `generation_statistics()`'s totals, and vice versa.
    """
    service = HistoryService(history_root=tmp_path / "history")
    service.save_index_history(
        _make_index_entry(feature="Appointments", estimated_embedding_cost=0.00044146), []
    )
    service.save_generation_history(
        _make_generation_draft(
            feature="Appointments",
            actual_usage=ActualUsage(
                model="gpt-5",
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                input_cost=MoneyAmount(usd=1.0, inr=87.0),
                output_cost=MoneyAmount(usd=0.5, inr=43.5),
                total_cost=MoneyAmount(usd=1.5, inr=130.5),
            ),
        )
    )

    stats = service.generation_statistics(_cost_calculator())
    [summary] = service.list_generation_summaries(_cost_calculator())

    assert stats.total_spend_usd == pytest.approx(1.5)
    assert stats.total_spend_inr == pytest.approx(130.5)
    assert summary.total_ai_cost_usd == pytest.approx(1.5)
    assert summary.total_ai_cost_inr == pytest.approx(130.5)

    # And the reverse: the generation's cost is never folded into indexing_statistics().
    indexing_stats = service.indexing_statistics(_cost_calculator())
    assert indexing_stats.total_embedding_cost_usd == pytest.approx(0.00044146)


# --- save_upload_history / list_upload_history ---


def test_save_upload_history_creates_a_timestamped_folder_under_upload_subfolder(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    folder = service.save_upload_history(_make_upload_entry(), [_make_preview_chunk()])

    assert folder.parent == tmp_path / "history" / "upload"
    assert _TIMESTAMP_PATTERN.fullmatch(folder.name)
    assert folder.is_dir()


def test_save_upload_history_does_not_touch_index_history(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)
    service.save_index_history(_make_index_entry(), [])
    index_before = {p.name for p in (history_root / "index").iterdir()}

    service.save_upload_history(_make_upload_entry(), [])

    assert {p.name for p in (history_root / "index").iterdir()} == index_before
    assert not (history_root / "upload" / "retrieval").exists()


def test_save_upload_history_writes_summary_with_all_required_fields(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    entry = _make_upload_entry(upload_session_id="sess-xyz", chunks_indexed=7)

    folder = service.save_upload_history(entry, [_make_preview_chunk()])

    summary = json.loads((folder / "uploaded_embedding_summary.json").read_text(encoding="utf-8"))
    assert summary["upload_session_id"] == "sess-xyz"
    assert summary["uploaded_at"]
    assert summary["embedding_model"] == "text-embedding-3-small"
    assert summary["documents_indexed"] == 1
    assert summary["chunks_indexed"] == 7
    assert summary["embedding_tokens"] == 10
    assert summary["average_tokens_per_chunk"] == entry.average_tokens_per_chunk
    assert summary["estimated_embedding_cost"] == entry.estimated_embedding_cost
    assert summary["elapsed_seconds"] == entry.elapsed_seconds
    assert summary["chroma_collection_name"] == "uploaded_documents_sess-xyz"
    assert summary["index_status"] == "SUCCESS"


def test_save_upload_history_writes_embedding_preview_in_the_same_format_as_index_history(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    preview = [_make_preview_chunk(chunk_id="chunk-1", page_number=2)]

    folder = service.save_upload_history(_make_upload_entry(), preview)

    [chunk] = json.loads((folder / "embedding_preview.json").read_text(encoding="utf-8"))
    assert chunk["chunk_id"] == "chunk-1"
    assert chunk["page_number"] == 2
    assert "embedding" not in chunk


def test_list_upload_history_returns_full_records_for_every_saved_run(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    service.save_upload_history(_make_upload_entry(upload_session_id="sess-a"), [_make_preview_chunk()])
    service.save_upload_history(_make_upload_entry(upload_session_id="sess-b"), [])

    records = service.list_upload_history()

    assert {record.summary.upload_session_id for record in records} == {"sess-a", "sess-b"}
    by_session = {record.summary.upload_session_id: record for record in records}
    assert len(by_session["sess-a"].embedding_preview) == 1
    assert by_session["sess-a"].history_id  # a real folder name, not empty


def test_list_upload_history_returns_empty_list_when_nothing_saved(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    assert service.list_upload_history() == []


def test_list_upload_history_ignores_index_only_folders(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    service.save_index_history(_make_index_entry(), [])

    assert service.list_upload_history() == []


def test_history_service_is_reusable_for_a_future_history_subtype(tmp_path):
    """Not a real feature yet — proves the generic timestamped-folder
    helper HistoryService uses for indexing doesn't hardcode "index" in a
    way that would block a future save_retrieval_history/
    save_generation_history from reusing the same mechanism, and that a
    sibling subtype folder can coexist without HistoryService's index
    methods getting confused by it.
    """
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)
    service.save_index_history(_make_index_entry(feature="Appointments"), [])

    hypothetical_retrieval_folder = service._create_timestamped_folder(history_root / "retrieval")

    assert hypothetical_retrieval_folder.parent == history_root / "retrieval"
    assert _TIMESTAMP_PATTERN.fullmatch(hypothetical_retrieval_folder.name)
    # The sibling subtype folder doesn't leak into or confuse indexing reads.
    entries = service.list_index_history()
    assert len(entries) == 1
    assert entries[0].feature == "Appointments"


# --- save_generation_history / list_generations / load_generation / delete_generation / generation_statistics ---


def test_save_generation_history_writes_a_single_json_file_under_generation_subfolder(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)

    entry = service.save_generation_history(_make_generation_draft())

    generation_file = history_root / "generation" / f"{entry.generation_id}.json"
    assert generation_file.is_file()
    assert _GENERATION_TIMESTAMP_PATTERN.fullmatch(entry.generation_id)
    # Exactly one file for this generation — no sibling folder of files
    # like index/upload history produce.
    assert list((history_root / "generation").iterdir()) == [generation_file]


def test_save_generation_history_assigns_generation_id_and_timestamp_from_the_same_moment(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    entry = service.save_generation_history(_make_generation_draft())

    assert entry.generation_id
    assert entry.timestamp
    assert entry.timestamp.strftime("%Y-%m-%dT%H-%M-%S") == entry.generation_id.split("_")[0]


def test_save_generation_history_does_not_touch_index_or_upload_history(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)
    service.save_index_history(_make_index_entry(), [])
    service.save_upload_history(_make_upload_entry(), [])
    index_before = {p.name for p in (history_root / "index").iterdir()}
    upload_before = {p.name for p in (history_root / "upload").iterdir()}

    service.save_generation_history(_make_generation_draft())

    assert {p.name for p in (history_root / "index").iterdir()} == index_before
    assert {p.name for p in (history_root / "upload").iterdir()} == upload_before


def test_save_generation_history_writes_json_with_all_required_fields(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)
    draft = _make_generation_draft(feature="Appointments", redmine_ticket="12345")

    entry = service.save_generation_history(draft)

    raw = json.loads((history_root / "generation" / f"{entry.generation_id}.json").read_text(encoding="utf-8"))
    assert raw["generation_id"] == entry.generation_id
    assert raw["timestamp"]
    assert raw["feature"] == "Appointments"
    assert raw["redmine_ticket"] == "12345"
    assert raw["model"] == "gpt-5"
    assert raw["prompt_version"] == "1.0.0"
    assert raw["retrieval"] == {
        "workflow_chunks": 1,
        "historical_test_cases": 1,
        "historical_issues": 1,
        "uploaded_documents": 0,
    }
    assert raw["estimated_usage"]["input_tokens"] == 1000
    assert raw["estimated_usage"]["estimated_input_cost"] == {"usd": 0.00125, "inr": 0.11}
    assert raw["actual_usage"]["prompt_tokens"] == 1000
    assert raw["actual_usage"]["completion_tokens"] == 200
    assert raw["actual_usage"]["total_cost"] == {"usd": 0.00325, "inr": 0.28}
    assert raw["pricing"] == {
        "model": "gpt-5",
        "input_price_per_million_tokens": 1.25,
        "output_price_per_million_tokens": 10.00,
        "usd_to_inr_exchange_rate": 87.00,
    }
    assert raw["generation_time_ms"] == 2300.0
    assert raw["prompt"] == draft.prompt
    assert raw["response"] == draft.response
    assert len(raw["test_cases"]) == 1
    assert raw["test_cases"][0]["test_case_id"] == "TC-1"
    assert raw["test_cases"][0]["steps"] == [
        {"step_no": 1, "action": "Open appointment.", "expected_result": "Details shown."}
    ]
    assert raw["metadata"] == {
        "top_k": 5,
        "upload_session_id": None,
        "generated_test_cases": 1,
        "requested_test_cases": None,
        "count_target_met": None,
        "coverage_note": None,
    }
    assert raw["status"] == "SUCCESS"


def test_save_generation_history_does_not_collide_on_repeated_calls(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    first = service.save_generation_history(_make_generation_draft())
    second = service.save_generation_history(_make_generation_draft())

    assert first.generation_id != second.generation_id


def test_list_generations_returns_every_saved_entry(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    service.save_generation_history(_make_generation_draft(feature="Appointments"))
    service.save_generation_history(_make_generation_draft(feature="Coding Tool"))

    entries = service.list_generations()

    assert {entry.feature for entry in entries} == {"Appointments", "Coding Tool"}


def test_list_generations_returns_empty_list_when_nothing_saved(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    assert service.list_generations() == []


def test_list_generations_skips_a_corrupted_file_without_failing(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)
    service.save_generation_history(_make_generation_draft(feature="Appointments"))
    generation_root = history_root / "generation"
    (generation_root / "corrupted.json").write_text("{not valid json", encoding="utf-8")
    (generation_root / "wrong-schema.json").write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")

    entries = service.list_generations()

    assert len(entries) == 1
    assert entries[0].feature == "Appointments"


def test_load_generation_returns_the_full_entry(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    saved = service.save_generation_history(_make_generation_draft(feature="Appointments"))

    loaded = service.load_generation(saved.generation_id)

    assert loaded == saved
    assert loaded.pricing.model == "gpt-5"
    assert loaded.metadata.generated_test_cases == 1


def test_load_generation_raises_not_found_for_unknown_id(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    with pytest.raises(NotFoundError):
        service.load_generation("2000-01-01T00-00-00")


def test_load_generation_raises_external_service_error_for_a_corrupted_file(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)
    generation_root = history_root / "generation"
    generation_root.mkdir(parents=True)
    (generation_root / "2026-01-01T00-00-00.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ExternalServiceError):
        service.load_generation("2026-01-01T00-00-00")


def test_load_generation_raises_external_service_error_for_a_schema_mismatched_file(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)
    generation_root = history_root / "generation"
    generation_root.mkdir(parents=True)
    (generation_root / "2026-01-01T00-00-00.json").write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")

    with pytest.raises(ExternalServiceError):
        service.load_generation("2026-01-01T00-00-00")


# --- delete_generation ---


def test_delete_generation_removes_only_that_generations_file(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    keep = service.save_generation_history(_make_generation_draft(feature="Appointments"))
    remove = service.save_generation_history(_make_generation_draft(feature="Coding Tool"))

    service.delete_generation(remove.generation_id)

    remaining = service.list_generations()
    assert [entry.generation_id for entry in remaining] == [keep.generation_id]
    with pytest.raises(NotFoundError):
        service.load_generation(remove.generation_id)


def test_delete_generation_does_not_touch_index_or_upload_history(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)
    service.save_index_history(_make_index_entry(), [])
    service.save_upload_history(_make_upload_entry(), [])
    index_before = {p.name for p in (history_root / "index").iterdir()}
    upload_before = {p.name for p in (history_root / "upload").iterdir()}
    saved = service.save_generation_history(_make_generation_draft())

    service.delete_generation(saved.generation_id)

    assert {p.name for p in (history_root / "index").iterdir()} == index_before
    assert {p.name for p in (history_root / "upload").iterdir()} == upload_before


def test_delete_generation_raises_not_found_for_unknown_id(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    with pytest.raises(NotFoundError):
        service.delete_generation("2000-01-01T00-00-00")


# --- generation_statistics ---


def test_generation_statistics_returns_zeroed_stats_when_nothing_saved(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")

    stats = service.generation_statistics(_cost_calculator())

    assert stats.total_generations == 0
    assert stats.successful_generations == 0
    assert stats.failed_generations == 0
    assert stats.average_generation_time_ms == 0.0
    assert stats.average_test_cases == 0.0
    assert stats.average_prompt_tokens == 0.0
    assert stats.average_completion_tokens == 0.0
    assert stats.average_cost_usd == 0.0
    assert stats.average_cost_inr == 0.0
    assert stats.most_used_feature is None
    assert stats.most_used_model is None
    assert stats.total_prompt_tokens == 0
    assert stats.total_completion_tokens == 0
    assert stats.total_tokens == 0
    assert stats.total_spend_usd == 0.0
    assert stats.total_spend_inr == 0.0


def test_generation_statistics_computes_totals_and_averages(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    service.save_generation_history(
        _make_generation_draft(
            feature="Appointments",
            actual_usage=ActualUsage(
                model="gpt-5",
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                input_cost=MoneyAmount(usd=1.0, inr=87.0),
                output_cost=MoneyAmount(usd=0.5, inr=43.5),
                total_cost=MoneyAmount(usd=1.5, inr=130.5),
            ),
            generation_time_ms=1000.0,
        )
    )
    service.save_generation_history(
        _make_generation_draft(
            feature="Appointments",
            actual_usage=ActualUsage(
                model="gpt-5",
                prompt_tokens=3000,
                completion_tokens=600,
                total_tokens=3600,
                input_cost=MoneyAmount(usd=3.0, inr=261.0),
                output_cost=MoneyAmount(usd=1.5, inr=130.5),
                total_cost=MoneyAmount(usd=4.5, inr=391.5),
            ),
            generation_time_ms=3000.0,
        )
    )

    stats = service.generation_statistics(_cost_calculator())

    assert stats.total_generations == 2
    assert stats.successful_generations == 2
    assert stats.failed_generations == 0
    assert stats.average_generation_time_ms == 2000.0
    assert stats.average_test_cases == 1.0
    assert stats.average_prompt_tokens == 2000.0
    assert stats.average_completion_tokens == 400.0
    assert stats.average_cost_usd == 3.0
    assert stats.average_cost_inr == pytest.approx(261.0)
    assert stats.most_used_feature == "Appointments"
    assert stats.most_used_model == "gpt-5"
    assert stats.total_prompt_tokens == 4000
    assert stats.total_completion_tokens == 800
    assert stats.total_tokens == 4800
    assert stats.total_spend_usd == 6.0
    assert stats.total_spend_inr == pytest.approx(522.0)


def test_generation_statistics_reports_the_most_common_feature_and_model(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    service.save_generation_history(_make_generation_draft(feature="Appointments"))
    service.save_generation_history(_make_generation_draft(feature="Appointments"))
    service.save_generation_history(_make_generation_draft(feature="Coding Tool"))

    stats = service.generation_statistics(_cost_calculator())

    assert stats.most_used_feature == "Appointments"


def test_generation_statistics_excludes_corrupted_files(tmp_path):
    history_root = tmp_path / "history"
    service = HistoryService(history_root=history_root)
    service.save_generation_history(_make_generation_draft(feature="Appointments"))
    (history_root / "generation" / "corrupted.json").write_text("{not valid json", encoding="utf-8")

    stats = service.generation_statistics(_cost_calculator())

    assert stats.total_generations == 1


# --- list_generation_summaries (Embedding-stage join) ---


def test_list_generation_summaries_attaches_embedding_cost_for_matching_upload_session(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    service.save_upload_history(
        _make_upload_entry(upload_session_id="sess-abc123", embedding_tokens=500, estimated_embedding_cost=0.00001),
        [],
    )
    service.save_generation_history(
        _make_generation_draft(
            metadata=GenerationMetadata(top_k=5, upload_session_id="sess-abc123", generated_test_cases=1)
        )
    )

    [summary] = service.list_generation_summaries(_cost_calculator())

    assert summary.embedding_tokens == 500
    assert summary.embedding_cost_usd == pytest.approx(0.00001)
    assert summary.embedding_cost_inr == pytest.approx(round(0.00001 * 87.00, 2))
    assert summary.total_ai_cost_usd == pytest.approx(summary.embedding_cost_usd + summary.actual_generation_cost_usd)
    assert summary.total_ai_cost_inr == pytest.approx(summary.embedding_cost_inr + summary.actual_generation_cost_inr)


def test_list_generation_summaries_leaves_embedding_fields_none_without_upload_session(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    service.save_generation_history(
        _make_generation_draft(metadata=GenerationMetadata(top_k=5, upload_session_id=None, generated_test_cases=1))
    )

    [summary] = service.list_generation_summaries(_cost_calculator())

    assert summary.embedding_tokens is None
    assert summary.embedding_cost_usd is None
    assert summary.embedding_cost_inr is None
    assert summary.total_ai_cost_usd == summary.actual_generation_cost_usd


def test_list_generation_summaries_leaves_embedding_fields_none_when_session_never_embedded(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    service.save_generation_history(
        _make_generation_draft(
            metadata=GenerationMetadata(top_k=5, upload_session_id="sess-never-embedded", generated_test_cases=1)
        )
    )

    [summary] = service.list_generation_summaries(_cost_calculator())

    assert summary.embedding_tokens is None


def test_list_generation_summaries_uses_the_latest_upload_run_for_a_session(tmp_path):
    service = HistoryService(history_root=tmp_path / "history")
    older = utcnow()
    service.save_upload_history(
        _make_upload_entry(upload_session_id="sess-abc123", uploaded_at=older, embedding_tokens=100),
        [],
    )
    service.save_upload_history(
        _make_upload_entry(
            upload_session_id="sess-abc123",
            uploaded_at=older.replace(year=older.year + 1),
            embedding_tokens=999,
        ),
        [],
    )
    service.save_generation_history(
        _make_generation_draft(
            metadata=GenerationMetadata(top_k=5, upload_session_id="sess-abc123", generated_test_cases=1)
        )
    )

    [summary] = service.list_generation_summaries(_cost_calculator())

    assert summary.embedding_tokens == 999
