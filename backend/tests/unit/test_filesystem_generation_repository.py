"""Unit tests for `FileSystemGenerationRepository` — the filesystem
implementation of `GenerationRepository`. Every test uses `tmp_path`, so
none of these ever read or write the real `backend/database/`.
"""
import json

import pytest

from app.core.exceptions import ExternalServiceError, NotFoundError
from app.models.generation_record import GenerationFeedbackEvaluation, RetrievedChunkRef
from app.repositories.filesystem_generation_repository import FileSystemGenerationRepository


def _repository(tmp_path) -> FileSystemGenerationRepository:
    return FileSystemGenerationRepository(database_root=tmp_path / "database")


# --- create_generation / directory creation ---


def test_create_generation_creates_the_generation_directory(tmp_path):
    repository = _repository(tmp_path)

    repository.create_generation("gen_001")

    assert (tmp_path / "database" / "generations" / "gen_001").is_dir()


def test_create_generation_is_safe_to_call_more_than_once(tmp_path):
    repository = _repository(tmp_path)

    repository.create_generation("gen_001")
    repository.create_generation("gen_001")

    assert (tmp_path / "database" / "generations" / "gen_001").is_dir()


def test_save_user_question_creates_the_generation_directory_without_create_generation(tmp_path):
    """Section 12: it should not be necessary to manually create the
    directory before persisting a record."""
    repository = _repository(tmp_path)

    repository.save_user_question("gen_001", "How does Contact Log work?")

    assert (tmp_path / "database" / "generations" / "gen_001").is_dir()


# --- save_user_question ---


def test_save_user_question_persists_required_fields(tmp_path):
    repository = _repository(tmp_path)

    record = repository.save_user_question("gen_001", "How does Contact Log work?")

    assert record.generation_id == "gen_001"
    assert record.user_question == "How does Contact Log work?"
    assert record.created_at is not None

    path = tmp_path / "database" / "generations" / "gen_001" / "user_question.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["generation_id"] == "gen_001"
    assert raw["user_question"] == "How does Contact Log work?"
    assert raw["created_at"]


# --- save_retrieved_chunks ---


def test_save_retrieved_chunks_persists_chunks_and_documents_without_embeddings(tmp_path):
    repository = _repository(tmp_path)
    chunks = [
        RetrievedChunkRef(
            chunk_id="chunk_123", document_id="doc_10", document_name="Contact Workflow.pdf", similarity_score=0.91
        )
    ]

    record = repository.save_retrieved_chunks(
        "gen_001",
        chunks_retrieved=chunks,
        documents_retrieved=["Contact Workflow.pdf", "Contact Log Test Cases.xlsx"],
    )

    assert record.generation_id == "gen_001"
    assert record.chunks_retrieved == chunks
    assert record.documents_retrieved == ["Contact Workflow.pdf", "Contact Log Test Cases.xlsx"]

    path = tmp_path / "database" / "generations" / "gen_001" / "retrieved_chunks.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    [persisted_chunk] = raw["chunks_retrieved"]
    assert persisted_chunk["chunk_id"] == "chunk_123"
    assert persisted_chunk["document_id"] == "doc_10"
    assert persisted_chunk["document_name"] == "Contact Workflow.pdf"
    assert persisted_chunk["similarity_score"] == 0.91
    # No embedding vectors — ChromaDB remains the only store for those.
    assert "embedding" not in json.dumps(raw).lower()


# --- save_prompt ---


def test_save_prompt_persists_prompt_and_version(tmp_path):
    repository = _repository(tmp_path)

    record = repository.save_prompt(
        "gen_001",
        prompt="Answer the question using only the provided context.",
        input_tokens=3200,
        estimated_tokens=3500,
        prompt_version="knowledge-assistant-v1",
    )

    assert record.prompt == "Answer the question using only the provided context."
    assert record.input_tokens == 3200
    assert record.estimated_tokens == 3500
    assert record.prompt_version == "knowledge-assistant-v1"

    path = tmp_path / "database" / "generations" / "gen_001" / "prompt.json"
    assert json.loads(path.read_text(encoding="utf-8"))["prompt_version"] == "knowledge-assistant-v1"


# --- save_response ---


def test_save_response_persists_answer_and_output_tokens(tmp_path):
    repository = _repository(tmp_path)

    record = repository.save_response("gen_001", response_answer="Contact Log is...", output_tokens=580)

    assert record.response_answer == "Contact Log is..."
    assert record.output_tokens == 580

    path = tmp_path / "database" / "generations" / "gen_001" / "response.json"
    assert json.loads(path.read_text(encoding="utf-8"))["response_answer"] == "Contact Log is..."


def test_save_response_persists_source_documents(tmp_path):
    repository = _repository(tmp_path)

    record = repository.save_response(
        "gen_001",
        response_answer="Contact Log is...",
        output_tokens=580,
        source_documents=["Contact_Log_Workflow.pdf", "Contact_Log_TestCases.xlsx"],
    )

    assert record.source_documents == ["Contact_Log_Workflow.pdf", "Contact_Log_TestCases.xlsx"]

    generation = repository.get_generation("gen_001")
    assert generation.response.source_documents == ["Contact_Log_Workflow.pdf", "Contact_Log_TestCases.xlsx"]


def test_save_response_defaults_source_documents_to_an_empty_list(tmp_path):
    """Backward compatibility: a caller that doesn't track source
    documents (or a `response.json` persisted before this field
    existed) never fails to load — it just has none."""
    repository = _repository(tmp_path)

    record = repository.save_response("gen_001", response_answer="Contact Log is...", output_tokens=580)

    assert record.source_documents == []


# --- save_feedback ---


def test_save_feedback_persists_a_good_evaluation_with_no_description(tmp_path):
    repository = _repository(tmp_path)

    record = repository.save_feedback("gen_001", evaluation=GenerationFeedbackEvaluation.GOOD)

    assert record.evaluation == GenerationFeedbackEvaluation.GOOD
    assert record.description is None

    path = tmp_path / "database" / "generations" / "gen_001" / "feedback.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["evaluation"] == "GOOD"
    assert raw["description"] is None


def test_save_feedback_persists_a_bad_evaluation_with_a_description(tmp_path):
    repository = _repository(tmp_path)

    record = repository.save_feedback(
        "gen_001",
        evaluation=GenerationFeedbackEvaluation.BAD,
        description="The answer missed the Bridge functionality.",
    )

    assert record.evaluation == GenerationFeedbackEvaluation.BAD
    assert record.description == "The answer missed the Bridge functionality."


# --- save_usage ---


def test_save_usage_persists_tokens_cost_and_timing_exactly_as_given(tmp_path):
    repository = _repository(tmp_path)

    record = repository.save_usage(
        "gen_001",
        model="gpt-5",
        input_tokens=3200,
        output_tokens=580,
        total_tokens=3780,
        estimated_input_tokens=3500,
        input_cost_inr=0.40,
        output_cost_inr=0.55,
        total_cost_inr=0.95,
        generation_time_ms=8420,
    )

    assert record.model == "gpt-5"
    assert record.total_tokens == 3780
    assert record.total_cost_inr == 0.95
    assert record.generation_time_ms == 8420

    path = tmp_path / "database" / "generations" / "gen_001" / "usage.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["total_tokens"] == 3780


# --- generation_id consistency ---


def test_the_same_generation_id_is_used_across_every_record_type(tmp_path):
    """The repository never invents a second id — every record for one
    generation carries exactly the id the caller supplied."""
    repository = _repository(tmp_path)
    generation_id = "gen_20260812_abc123"

    repository.save_user_question(generation_id, "How does Contact Log work?")
    repository.save_retrieved_chunks(generation_id, [], [])
    repository.save_prompt(generation_id, "prompt text", 100, 120, "knowledge-assistant-v1")
    repository.save_response(generation_id, "answer text", 50)
    repository.save_feedback(generation_id, GenerationFeedbackEvaluation.GOOD)
    repository.save_usage(generation_id, "gpt-5", 100, 50, 150, 120, 0.1, 0.1, 0.2, 500.0)

    generation = repository.get_generation(generation_id)

    assert generation.generation_id == generation_id
    assert generation.user_question.generation_id == generation_id
    assert generation.retrieved_chunks.generation_id == generation_id
    assert generation.prompt.generation_id == generation_id
    assert generation.response.generation_id == generation_id
    assert generation.feedback.generation_id == generation_id
    assert generation.usage.generation_id == generation_id

    # Exactly one directory was created — no per-record-type id/folder invented.
    assert list((tmp_path / "database" / "generations").iterdir()) == [
        tmp_path / "database" / "generations" / generation_id
    ]


# --- get_generation (reconstruction) ---


def test_get_generation_reconstructs_every_persisted_record(tmp_path):
    repository = _repository(tmp_path)
    repository.save_user_question("gen_001", "How does Contact Log work?")
    repository.save_retrieved_chunks("gen_001", [], ["Contact Workflow.pdf"])
    repository.save_prompt("gen_001", "prompt text", 100, 120, "knowledge-assistant-v1")
    repository.save_response("gen_001", "Contact Log is...", 50)
    repository.save_feedback("gen_001", GenerationFeedbackEvaluation.GOOD)
    repository.save_usage("gen_001", "gpt-5", 100, 50, 150, 120, 0.1, 0.1, 0.2, 500.0)

    generation = repository.get_generation("gen_001")

    assert generation.user_question.user_question == "How does Contact Log work?"
    assert generation.retrieved_chunks.documents_retrieved == ["Contact Workflow.pdf"]
    assert generation.prompt.prompt_version == "knowledge-assistant-v1"
    assert generation.response.response_answer == "Contact Log is..."
    assert generation.feedback.evaluation == GenerationFeedbackEvaluation.GOOD
    assert generation.usage.model == "gpt-5"


def test_get_generation_raises_not_found_for_an_unknown_generation_id(tmp_path):
    repository = _repository(tmp_path)

    with pytest.raises(NotFoundError):
        repository.get_generation("gen_does_not_exist")


def test_get_generation_raises_external_service_error_for_a_corrupted_record(tmp_path):
    repository = _repository(tmp_path)
    repository.save_user_question("gen_001", "How does Contact Log work?")
    prompt_path = tmp_path / "database" / "generations" / "gen_001" / "prompt.json"
    prompt_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ExternalServiceError):
        repository.get_generation("gen_001")


# --- missing feedback handling ---


def test_get_generation_returns_none_feedback_when_no_feedback_was_submitted(tmp_path):
    """Section 13: a generation may exist before feedback is submitted
    — this must not be treated as an error."""
    repository = _repository(tmp_path)
    repository.save_user_question("gen_001", "How does Contact Log work?")
    repository.save_response("gen_001", "Contact Log is...", 50)

    generation = repository.get_generation("gen_001")

    assert generation.feedback is None
    assert generation.user_question is not None
    assert generation.response is not None


def test_get_generation_returns_none_for_every_record_not_yet_saved(tmp_path):
    """A generation may exist in an even earlier, partially-saved state
    than "everything but feedback" — e.g. right after
    `create_generation`, before any save_* call."""
    repository = _repository(tmp_path)
    repository.create_generation("gen_001")

    generation = repository.get_generation("gen_001")

    assert generation.user_question is None
    assert generation.retrieved_chunks is None
    assert generation.prompt is None
    assert generation.response is None
    assert generation.feedback is None
    assert generation.usage is None


# --- list_generations ---


def test_list_generations_returns_empty_list_when_nothing_saved(tmp_path):
    repository = _repository(tmp_path)

    assert repository.list_generations() == []


def test_list_generations_returns_every_generation(tmp_path):
    repository = _repository(tmp_path)
    repository.save_user_question("gen_001", "Question one")
    repository.save_user_question("gen_002", "Question two")

    items = repository.list_generations()

    assert {item.generation_id for item in items} == {"gen_001", "gen_002"}
    assert {item.user_question for item in items} == {"Question one", "Question two"}


def test_list_generations_sorts_newest_first(tmp_path):
    repository = _repository(tmp_path)
    repository.save_user_question("gen_001", "Oldest question")
    first = repository.get_generation("gen_001").user_question.created_at

    # Force a distinct, later `created_at` rather than depending on real
    # wall-clock timing between two fast, back-to-back calls.
    second_path = tmp_path / "database" / "generations" / "gen_002" / "user_question.json"
    second_path.parent.mkdir(parents=True)
    second_path.write_text(
        json.dumps(
            {
                "generation_id": "gen_002",
                "user_question": "Newest question",
                "created_at": first.replace(year=first.year + 1).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    items = repository.list_generations()

    assert [item.generation_id for item in items] == ["gen_002", "gen_001"]


def test_list_generations_sorts_items_with_an_unreadable_user_question_last(tmp_path):
    repository = _repository(tmp_path)
    repository.save_user_question("gen_001", "A readable question")
    corrupted_dir = tmp_path / "database" / "generations" / "gen_002"
    corrupted_dir.mkdir(parents=True)
    (corrupted_dir / "user_question.json").write_text("{not valid json", encoding="utf-8")

    items = repository.list_generations()

    assert [item.generation_id for item in items] == ["gen_001", "gen_002"]
    assert items[1].user_question is None
    assert items[1].created_at is None


def test_list_generations_does_not_write_into_the_real_database_directory(tmp_path):
    """Sanity check that this repository is fully rooted at the
    injected `database_root` — never `backend/database/` itself."""
    repository = _repository(tmp_path)

    repository.save_user_question("gen_001", "A question")

    assert not (tmp_path / "generations").exists()


# --- list_usage_records ---


def test_list_usage_records_returns_empty_list_when_nothing_saved(tmp_path):
    repository = _repository(tmp_path)

    assert repository.list_usage_records() == []


def test_list_usage_records_returns_every_persisted_usage_record(tmp_path):
    repository = _repository(tmp_path)
    repository.save_usage("gen_001", "gpt-5", 100, 50, 150, 120, 0.1, 0.1, 0.2, 500.0)
    repository.save_usage("gen_002", "gpt-5", 200, 80, 280, 220, 0.2, 0.2, 0.4, 700.0)

    records = repository.list_usage_records()

    assert {record.generation_id for record in records} == {"gen_001", "gen_002"}
    assert {record.total_tokens for record in records} == {150, 280}


def test_list_usage_records_excludes_a_generation_with_no_usage_json(tmp_path):
    """A generation that never completed successfully (e.g. it failed
    before `save_usage` was ever called) has no `usage.json` — it must
    simply not appear, not be treated as an error."""
    repository = _repository(tmp_path)
    repository.save_user_question("gen_001", "Question with no usage yet")
    repository.save_usage("gen_002", "gpt-5", 100, 50, 150, 120, 0.1, 0.1, 0.2, 500.0)

    records = repository.list_usage_records()

    assert [record.generation_id for record in records] == ["gen_002"]


def test_list_usage_records_skips_a_corrupted_usage_json_without_raising(tmp_path, caplog):
    """One bad `usage.json` must not break the whole listing — it's
    skipped and logged, per the Usage Dashboard's error-handling
    requirement that a corrupt Knowledge Assistant record never crashes
    the dashboard."""
    repository = _repository(tmp_path)
    repository.save_usage("gen_001", "gpt-5", 100, 50, 150, 120, 0.1, 0.1, 0.2, 500.0)
    corrupted_usage_path = tmp_path / "database" / "generations" / "gen_002" / "usage.json"
    corrupted_usage_path.parent.mkdir(parents=True)
    corrupted_usage_path.write_text("{not valid json", encoding="utf-8")

    with caplog.at_level("WARNING"):
        records = repository.list_usage_records()

    assert [record.generation_id for record in records] == ["gen_001"]
    assert any("gen_002" in message for message in caplog.messages)


def test_list_usage_records_returns_empty_list_when_generations_root_does_not_exist(tmp_path):
    repository = _repository(tmp_path)

    assert repository.list_usage_records() == []
