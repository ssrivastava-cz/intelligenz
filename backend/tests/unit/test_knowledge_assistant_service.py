"""Unit tests for `KnowledgeAssistantService` — orchestration only:
generate generation_id -> RetrievalService -> KnowledgeAssistantPromptBuilder
-> local token count -> persist question/chunks/prompt -> OpenAI Chat
(exactly once) -> validate -> persist response/usage, with `error.json`
persisted (tagged with the stage that failed) whenever any step raises.
Every dependency here is real except OpenAI (the one boundary faked, per
tests/fakes.py) — the same `FakeOpenAIClient` instance stands in for
both the embedding call `RetrievalService` needs and the chat completion
this service makes, exactly like `tests/unit/test_generation_service.py`
does for the (separate, unmodified) Test Plan Generator. ChromaDB is a
real `chromadb.EphemeralClient`; persistence is the real
`FileSystemGenerationRepository` rooted at `tmp_path`.
"""
import json
import re
import uuid

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.core.exceptions import ExternalServiceError, ValidationError
from app.models.generation_record import KnowledgeAssistantErrorStage
from app.repositories.filesystem_generation_repository import FileSystemGenerationRepository
from app.retrievers.text_tokenizer import TOKENIZER_VERSION, tokenize
from app.services.bm25_index_manager import BM25IndexManager
from app.services.cost_calculator import CostCalculator
from app.services.embedding_service import EmbeddingService
from app.services.knowledge_assistant_prompt_builder import KnowledgeAssistantPromptBuilder
from app.services.knowledge_assistant_service import KnowledgeAssistantService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store_service import VectorStoreService
from tests.fakes import FakeOpenAIClient

_EMBEDDING_MODEL = "text-embedding-3-small"
_CHAT_MODEL = "gpt-5"
_INPUT_PRICE_PER_MILLION = 1.25
_OUTPUT_PRICE_PER_MILLION = 10.00
_USD_TO_INR_EXCHANGE_RATE = 87.0
_GENERATION_ID_PATTERN = re.compile(r"^gen_\d{8}_[0-9a-f]{5}$")

_VALID_ANSWER_NO_SOURCES = json.dumps({"answer": "The documentation does not cover this.", "sourceDocuments": []})


def _valid_answer(
    source_documents: list[str] | None = None, answer: str = "Bridged contacts are merged automatically."
) -> str:
    return json.dumps({"answer": answer, "sourceDocuments": source_documents or []})


class _BrokenPromptBuilder:
    """Stands in for `KnowledgeAssistantPromptBuilder` to exercise the
    PROMPT_BUILDING failure path — the real builder is a pure formatter
    that essentially can't fail on well-formed input.
    """

    def build(self, *args, **kwargs):
        raise RuntimeError("prompt builder exploded")


def _workflow_metadata(**overrides) -> dict:
    base = {
        "chunkId": "workflow-chunk",
        "artifactType": "WORKFLOW",
        "feature": "Contact Log",
        "documentSource": "source_of_truth",
        "sourceFilename": "Contact_Log_Workflow.pdf",
        "sectionHeading": "Bridged Contacts",
        "parserName": "PdfParser",
        "parserVersion": "1.0",
        "documentId": "doc-workflow-1",
        "chunkNumber": 4,
    }
    base.update(overrides)
    return base


def _make_service(tmp_path, fake_client: FakeOpenAIClient | None = None, prompt_builder=None):
    client = (
        fake_client
        if fake_client is not None
        else FakeOpenAIClient(dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES)
    )
    embedding_service = EmbeddingService(client=client, model=_EMBEDDING_MODEL)

    chroma_client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    # A unique collection name per service instance — `chromadb.EphemeralClient()`
    # instances share underlying storage process-wide, so a fixed name
    # here would collide with whatever another test in this file (or
    # another file entirely) already wrote under it.
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)

    _bm25_manager = BM25IndexManager(
        index_dir=tmp_path / "bm25",
        collection_name=collection_name,
        tokenizer=tokenize,
        tokenizer_version=TOKENIZER_VERSION,
        chunking_signature="chunk_size=500,chunk_overlap=50",
    )
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        chroma_client=chroma_client,
        bm25_index_manager=_bm25_manager,
        upload_collection_prefix="uploaded_documents",
        default_top_k=5,
    )
    cost_calculator = CostCalculator(
        model=_CHAT_MODEL,
        input_price_per_million_tokens=_INPUT_PRICE_PER_MILLION,
        output_price_per_million_tokens=_OUTPUT_PRICE_PER_MILLION,
        usd_to_inr_exchange_rate=_USD_TO_INR_EXCHANGE_RATE,
    )
    repository = FileSystemGenerationRepository(database_root=tmp_path / "database")

    service = KnowledgeAssistantService(
        retrieval_service=retrieval_service,
        prompt_builder=prompt_builder or KnowledgeAssistantPromptBuilder(),
        embedding_service=embedding_service,
        openai_client=client,
        cost_calculator=cost_calculator,
        generation_repository=repository,
    )
    return service, vector_store, client, repository


def _seed_workflow_chunk(vector_store: VectorStoreService, feature: str) -> None:
    vector_store.replace_feature_chunks(
        feature=feature,
        ids=["workflow-chunk"],
        embeddings=[[2.0, 2.0]],
        documents=["Bridged contacts are merged automatically when matching criteria align."],
        metadatas=[_workflow_metadata(feature=feature)],
    )


def _seed_chunk(
    vector_store: VectorStoreService,
    *,
    feature: str,
    chunk_id: str,
    source_filename: str,
    text: str,
    embedding: list[float] | None = None,
) -> None:
    """Like `_seed_workflow_chunk`, but for tests that need multiple,
    independently identifiable chunks — e.g. proving retrieval combines
    results indexed under different features.
    """
    vector_store.replace_feature_chunks(
        feature=feature,
        ids=[chunk_id],
        embeddings=[embedding or [2.0, 2.0]],
        documents=[text],
        metadatas=[
            _workflow_metadata(
                chunkId=chunk_id, feature=feature, sourceFilename=source_filename, documentId=f"doc-{chunk_id}"
            )
        ],
    )


def _ask(service, question: str = "How does Contact Log work?"):
    return service.ask(user_question=question)


# --- OpenAI: successful call, exactly one, correct prompt, correct model ---


def test_ask_returns_the_answer_from_a_successful_chat_completion(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_valid_answer())
    service, vector_store, _, _ = _make_service(tmp_path, fake_client)
    _seed_workflow_chunk(vector_store, feature="Contact Log")

    _, answer = _ask(service)

    assert answer.answer == "Bridged contacts are merged automatically."


def test_ask_returns_a_generation_id_in_the_gen_yyyymmdd_random_format(tmp_path):
    service, _, _, _ = _make_service(tmp_path)

    generation_id, _ = _ask(service)

    assert _GENERATION_ID_PATTERN.match(generation_id)


def test_ask_calls_openai_chat_exactly_once(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    _ask(service)

    assert len(fake_client.chat_calls) == 1


def test_ask_sends_exactly_the_prompt_knowledgeassistantpromptbuilder_built(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES)
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    persisted_prompt = repository.get_generation(generation_id).prompt.prompt
    [call] = fake_client.chat_calls
    assert call["messages"] == [{"role": "user", "content": persisted_prompt}]
    assert "USER QUESTION:\n\nHow does Contact Log work?" in call["messages"][0]["content"]


def test_ask_uses_the_model_configured_on_the_cost_calculator(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    _ask(service)

    [call] = fake_client.chat_calls
    assert call["model"] == _CHAT_MODEL


def test_ask_uses_structured_outputs_with_the_knowledge_assistant_answer_schema(tmp_path):
    from app.models.knowledge_assistant_answer import KnowledgeAssistantAnswer

    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    _ask(service)

    [call] = fake_client.chat_calls
    assert call["response_format"] is KnowledgeAssistantAnswer


# --- OpenAI: actual usage read from the response, never the local estimate ---


def test_ask_uses_actual_openai_usage_not_the_local_estimate(tmp_path):
    fake_client = FakeOpenAIClient(
        dimension=2,
        chat_response_content=_VALID_ANSWER_NO_SOURCES,
        chat_prompt_tokens=99_999,
        chat_completion_tokens=1_234,
    )
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    usage = repository.get_generation(generation_id).usage
    assert usage.input_tokens == 99_999
    assert usage.output_tokens == 1_234
    assert usage.total_tokens == 101_233
    # The local pre-generation estimate is a completely different (much
    # smaller) number for this short prompt — proof actual usage isn't
    # just a copy of the estimate.
    assert usage.estimated_input_tokens != usage.input_tokens


# --- OpenAI: failure handling ---


def test_openai_failure_raises_external_service_error(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_fail_with=RuntimeError("connection reset"))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ExternalServiceError):
        _ask(service)


def test_openai_failure_does_not_persist_a_response_or_usage_record(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_fail_with=RuntimeError("connection reset"))
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id = _ask_expecting_failure(service, ExternalServiceError)

    generation = repository.get_generation(generation_id)
    assert generation.response is None
    assert generation.usage is None
    # Question/retrieval/prompt were still persisted — the failure was
    # only in the OpenAI step, and those earlier steps are legitimate
    # audit records even when the call after them fails.
    assert generation.user_question is not None
    assert generation.prompt is not None


def test_openai_failure_does_not_retry(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_fail_with=RuntimeError("connection reset"))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ExternalServiceError):
        _ask(service)

    assert len(fake_client.chat_calls) == 1


# --- OpenAI: empty/invalid response rejected ---


def test_empty_response_is_rejected(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=None)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError):
        _ask(service)


def test_empty_answer_string_is_rejected(tmp_path):
    empty_answer = json.dumps({"answer": "   ", "sourceDocuments": []})
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=empty_answer)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError):
        _ask(service)


def test_malformed_json_response_is_rejected(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content="not valid json")
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError):
        _ask(service)


def test_a_refusal_is_rejected(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=None, chat_refusal="I can't help with that.")
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError):
        _ask(service)


def test_invalid_response_does_not_persist_a_response_or_usage_record(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content="not valid json")
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id = _ask_expecting_failure(service, ValidationError)

    generation = repository.get_generation(generation_id)
    assert generation.response is None
    assert generation.usage is None


# --- Response: model cannot cause an un-retrieved document to become a source ---


def test_a_source_document_that_was_never_retrieved_is_rejected(tmp_path):
    """Structured Outputs only constrains the *shape* of the response —
    this proves the service itself checks the *content* of
    `sourceDocuments` against what was actually retrieved."""
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_valid_answer(["Invented_Document.pdf"]))
    service, vector_store, _, _ = _make_service(tmp_path, fake_client)
    _seed_workflow_chunk(vector_store, feature="Contact Log")

    with pytest.raises(ValidationError):
        _ask(service)


def test_rejecting_an_invented_source_document_does_not_persist_response_or_usage(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_valid_answer(["Invented_Document.pdf"]))
    service, vector_store, _, repository = _make_service(tmp_path, fake_client)
    _seed_workflow_chunk(vector_store, feature="Contact Log")

    generation_id = _ask_expecting_failure(service, ValidationError)

    generation = repository.get_generation(generation_id)
    assert generation.response is None
    assert generation.usage is None


def test_a_source_document_that_was_actually_retrieved_is_accepted(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_valid_answer(["Contact_Log_Workflow.pdf"]))
    service, vector_store, _, _ = _make_service(tmp_path, fake_client)
    _seed_workflow_chunk(vector_store, feature="Contact Log")

    _, answer = _ask(service)

    assert answer.source_documents == ["Contact_Log_Workflow.pdf"]


# --- Response: persisted through GenerationRepository ---


def test_ask_persists_the_answer_to_response_json(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_valid_answer(answer="Here is the answer."))
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    generation = repository.get_generation(generation_id)
    assert generation.response.response_answer == "Here is the answer."


def test_ask_persists_output_tokens_to_response_json(tmp_path):
    fake_client = FakeOpenAIClient(
        dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES, chat_completion_tokens=77
    )
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    generation = repository.get_generation(generation_id)
    assert generation.response.output_tokens == 77


# --- Usage: actual tokens, cost via CostCalculator, generation time, estimated stays separate ---


def test_ask_computes_actual_cost_via_cost_calculator(tmp_path):
    fake_client = FakeOpenAIClient(
        dimension=2,
        chat_response_content=_VALID_ANSWER_NO_SOURCES,
        chat_prompt_tokens=1000,
        chat_completion_tokens=200,
    )
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    usage = repository.get_generation(generation_id).usage
    expected_input_cost = round((1000 / 1_000_000) * _INPUT_PRICE_PER_MILLION * _USD_TO_INR_EXCHANGE_RATE, 2)
    expected_output_cost = round((200 / 1_000_000) * _OUTPUT_PRICE_PER_MILLION * _USD_TO_INR_EXCHANGE_RATE, 2)
    assert usage.input_cost_inr == pytest.approx(expected_input_cost)
    assert usage.output_cost_inr == pytest.approx(expected_output_cost)
    assert usage.total_cost_inr == pytest.approx(round(expected_input_cost + expected_output_cost, 2))


def test_ask_persists_the_model_name(tmp_path):
    service, _, _, repository = _make_service(tmp_path)

    generation_id, _ = _ask(service)

    assert repository.get_generation(generation_id).usage.model == _CHAT_MODEL


def test_ask_persists_a_positive_generation_time(tmp_path):
    service, _, _, repository = _make_service(tmp_path)

    generation_id, _ = _ask(service)

    assert repository.get_generation(generation_id).usage.generation_time_ms > 0


def test_estimated_tokens_and_actual_tokens_remain_separate(tmp_path):
    fake_client = FakeOpenAIClient(
        dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES, chat_prompt_tokens=42, chat_completion_tokens=7
    )
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    generation = repository.get_generation(generation_id)
    # prompt.json's pre-call estimate is untouched by the actual OpenAI usage.
    assert generation.prompt.estimated_tokens > 0
    assert generation.prompt.estimated_tokens != 42
    # usage.json carries both, distinctly.
    assert generation.usage.input_tokens == 42
    assert generation.usage.estimated_input_tokens == generation.prompt.estimated_tokens


# --- Usage: OpenAI's detailed token breakdown (reasoning/cached) + local diagnostics ---
#
# Reproduces the reported bug: gen_20260813_9c771 persisted
# output_tokens=1094 for a visible answer of only ~60-100 words —
# because most of that figure was invisible reasoning tokens, not
# visible answer content.


def test_ask_persists_reasoning_tokens_from_the_openai_usage_detail(tmp_path):
    fake_client = FakeOpenAIClient(
        dimension=2,
        chat_response_content=_valid_answer(answer="Short visible answer."),
        chat_completion_tokens=1094,
        chat_reasoning_tokens=960,
    )
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    usage = repository.get_generation(generation_id).usage
    # The real, billed figure is never altered by the detailed breakdown.
    assert usage.output_tokens == 1094
    assert usage.reasoning_tokens == 960


def test_ask_persists_cached_tokens_from_the_openai_usage_detail(tmp_path):
    fake_client = FakeOpenAIClient(
        dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES, chat_cached_tokens=128
    )
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    assert repository.get_generation(generation_id).usage.cached_tokens == 128


def test_ask_persists_none_for_reasoning_and_cached_tokens_when_openai_omits_the_detail(tmp_path):
    """Not every model/response includes the detailed breakdown — `None`
    must mean "not tracked", never a fabricated `0`."""
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES)
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    usage = repository.get_generation(generation_id).usage
    assert usage.reasoning_tokens is None
    assert usage.cached_tokens is None


def test_ask_computes_visible_answer_tokens_locally_and_never_replaces_output_tokens(tmp_path):
    fake_client = FakeOpenAIClient(
        dimension=2,
        chat_response_content=_valid_answer(answer="Short visible answer."),
        chat_completion_tokens=1094,
        chat_reasoning_tokens=960,
    )
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    usage = repository.get_generation(generation_id).usage
    # A short visible answer tokenizes to a small number...
    assert 0 < usage.visible_answer_tokens < 20
    # ...which is nowhere near the real, billed output_tokens — proof
    # the local count never overwrites or is confused with it.
    assert usage.output_tokens == 1094
    assert usage.visible_answer_tokens != usage.output_tokens


def test_ask_computes_structured_output_overhead_as_the_remainder_after_reasoning_and_visible_tokens(tmp_path):
    fake_client = FakeOpenAIClient(
        dimension=2,
        chat_response_content=_valid_answer(answer="Short visible answer."),
        chat_completion_tokens=1094,
        chat_reasoning_tokens=960,
    )
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    usage = repository.get_generation(generation_id).usage
    expected_overhead = usage.output_tokens - usage.reasoning_tokens - usage.visible_answer_tokens
    assert usage.structured_output_overhead_tokens == expected_overhead
    assert usage.structured_output_overhead_tokens >= 0


def test_ask_floors_structured_output_overhead_at_zero(tmp_path):
    """If local tiktoken counting (a different tokenizer/approximation
    from OpenAI's own) ever puts reasoning + visible tokens above the
    real output_tokens, the overhead must clamp to 0, not go negative."""
    fake_client = FakeOpenAIClient(
        dimension=2,
        chat_response_content=_valid_answer(answer="Short visible answer."),
        chat_completion_tokens=10,
        chat_reasoning_tokens=9,
    )
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    assert repository.get_generation(generation_id).usage.structured_output_overhead_tokens >= 0


def test_ask_logs_the_complete_raw_openai_usage_object_at_debug_level(tmp_path, caplog):
    fake_client = FakeOpenAIClient(
        dimension=2,
        chat_response_content=_VALID_ANSWER_NO_SOURCES,
        chat_completion_tokens=1094,
        chat_reasoning_tokens=960,
    )
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with caplog.at_level("DEBUG"):
        _ask(service)

    [message] = [m for m in caplog.messages if "Raw OpenAI usage" in m]
    assert "1094" in message
    assert "960" in message


# --- Generation ID: generated once, same id across every persisted record ---


def test_ask_uses_the_same_generation_id_across_every_persisted_record(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_valid_answer(["Contact_Log_Workflow.pdf"]))
    service, vector_store, _, repository = _make_service(tmp_path, fake_client)
    _seed_workflow_chunk(vector_store, feature="Contact Log")

    generation_id, _ = _ask(service)

    generation = repository.get_generation(generation_id)
    assert generation.user_question.generation_id == generation_id
    assert generation.retrieved_chunks.generation_id == generation_id
    assert generation.prompt.generation_id == generation_id
    assert generation.response.generation_id == generation_id
    assert generation.usage.generation_id == generation_id


def test_ask_never_creates_more_than_one_generation_directory_per_call(tmp_path):
    service, _, _, repository = _make_service(tmp_path)

    generation_id, _ = _ask(service)

    assert list((tmp_path / "database" / "generations").iterdir()) == [
        tmp_path / "database" / "generations" / generation_id
    ]


# --- Input validation ---


def test_empty_question_is_rejected_before_any_retrieval_or_openai_call(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES)
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError):
        service.ask(user_question="   ")

    assert fake_client.calls == []
    assert fake_client.chat_calls == []


# --- retrieved chunks / prompt persistence (pre-existing coverage, still applies) ---


def test_ask_persists_retrieved_chunks_through_the_repository(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_valid_answer(["Contact_Log_Workflow.pdf"]))
    service, vector_store, _, repository = _make_service(tmp_path, fake_client)
    _seed_workflow_chunk(vector_store, feature="Contact Log")

    generation_id, _ = _ask(service)

    generation = repository.get_generation(generation_id)
    assert "Contact_Log_Workflow.pdf" in generation.retrieved_chunks.documents_retrieved
    [chunk] = generation.retrieved_chunks.chunks_retrieved
    assert chunk.chunk_id == "workflow-chunk"
    assert chunk.document_name == "Contact_Log_Workflow.pdf"
    assert chunk.similarity_score > 0


def test_ask_handles_no_retrieved_context_for_any_source(tmp_path):
    service, _, _, repository = _make_service(tmp_path)

    generation_id, _ = _ask(service, question="Is there documentation for this?")

    generation = repository.get_generation(generation_id)
    assert generation.retrieved_chunks.chunks_retrieved == []
    assert generation.retrieved_chunks.documents_retrieved == []
    assert generation.retrieved_chunks.workflow.candidate_retrieved == 0
    assert "No workflow documents were retrieved for this question." in generation.prompt.prompt


# --- Retrieval diagnostics: Hybrid Retrieval persisted via retrieve_hybrid ---


def test_ask_persists_retrieval_diagnostics(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_valid_answer(["Contact_Log_Workflow.pdf"]))
    service, vector_store, _, repository = _make_service(tmp_path, fake_client)
    _seed_workflow_chunk(vector_store, feature="Contact Log")

    generation_id, _ = _ask(service)

    diagnostics = repository.get_generation(generation_id).retrieval_diagnostics
    assert diagnostics is not None
    assert diagnostics.generation_id == generation_id
    assert diagnostics.diagnostics.final_chunk_count == 1
    assert diagnostics.diagnostics.vector_candidate_count >= 1


def test_ask_fills_in_estimated_prompt_tokens_on_the_persisted_diagnostics(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES)
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id, _ = _ask(service)

    generation = repository.get_generation(generation_id)
    assert generation.retrieval_diagnostics.diagnostics.estimated_prompt_tokens == generation.prompt.estimated_tokens


def test_ask_never_makes_more_openai_calls_because_of_hybrid_retrieval(tmp_path):
    """Section 11/12: BM25 and reranking must never call OpenAI — exactly
    one embedding call and one chat completion, same as before hybrid
    retrieval existed."""
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES)
    service, vector_store, _, _ = _make_service(tmp_path, fake_client)
    _seed_workflow_chunk(vector_store, feature="Contact Log")

    _ask(service)

    assert len(fake_client.calls) == 1
    assert len(fake_client.chat_calls) == 1


# --- Error persistence: retrieval / prompt building / OpenAI / response validation ---


def _ask_expecting_failure(service, expected_exception_type) -> str:
    """Asks a question expecting `ask()` to raise, and returns the
    `generation_id` from the raised exception's persisted trail (read
    back from whatever the service already wrote before failing) —
    `ask()` itself doesn't return an id on failure (it re-raises before
    reaching `return`), so tests recover it via the repository's
    newest-first `list_generations()` instead.
    """
    with pytest.raises(expected_exception_type):
        service.ask(user_question="How does Contact Log work?")
    [newest] = service._generation_repository.list_generations()[:1]
    return newest.generation_id


def test_retrieval_failure_never_calls_openai_chat(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, fail_with=ExternalServiceError("embedding request failed"))
    service, _, _, _ = _make_service(tmp_path, fake_client)

    with pytest.raises(ExternalServiceError):
        _ask(service)

    assert fake_client.chat_calls == []


def test_retrieval_failure_persists_an_error_record_with_the_retrieval_stage(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, fail_with=ExternalServiceError("embedding request failed"))
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id = _ask_expecting_failure(service, ExternalServiceError)

    generation = repository.get_generation(generation_id)
    assert generation.error is not None
    assert generation.error.stage == KnowledgeAssistantErrorStage.RETRIEVAL
    assert generation.error.error_type == "ExternalServiceError"
    assert "embedding request failed" in generation.error.error_message
    assert generation.response is None
    assert generation.usage is None


def test_prompt_building_failure_never_calls_openai_chat(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES)
    service, _, _, _ = _make_service(tmp_path, fake_client, prompt_builder=_BrokenPromptBuilder())

    with pytest.raises(RuntimeError):
        _ask(service)

    assert fake_client.chat_calls == []


def test_prompt_building_failure_persists_an_error_record_with_the_prompt_building_stage(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES)
    service, _, _, repository = _make_service(tmp_path, fake_client, prompt_builder=_BrokenPromptBuilder())

    generation_id = _ask_expecting_failure(service, RuntimeError)

    generation = repository.get_generation(generation_id)
    assert generation.error is not None
    assert generation.error.stage == KnowledgeAssistantErrorStage.PROMPT_BUILDING
    assert generation.error.error_type == "RuntimeError"
    assert "prompt builder exploded" in generation.error.error_message
    # Question and retrieval both succeeded before prompt building failed.
    assert generation.user_question is not None
    assert generation.retrieved_chunks is not None
    assert generation.prompt is None
    assert generation.response is None
    assert generation.usage is None


def test_openai_request_failure_persists_an_error_record_with_the_openai_request_stage(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_fail_with=RuntimeError("connection reset"))
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id = _ask_expecting_failure(service, ExternalServiceError)

    generation = repository.get_generation(generation_id)
    assert generation.error is not None
    assert generation.error.stage == KnowledgeAssistantErrorStage.OPENAI_REQUEST
    assert generation.error.error_type == "ExternalServiceError"
    assert generation.response is None
    assert generation.usage is None


def test_response_validation_failure_persists_an_error_record_with_the_response_validation_stage(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content="not valid json")
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id = _ask_expecting_failure(service, ValidationError)

    generation = repository.get_generation(generation_id)
    assert generation.error is not None
    assert generation.error.stage == KnowledgeAssistantErrorStage.RESPONSE_VALIDATION
    assert generation.error.error_type == "ValidationError"
    assert generation.response is None
    assert generation.usage is None


def test_an_invented_source_document_failure_is_tagged_with_the_response_validation_stage(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_valid_answer(["Invented_Document.pdf"]))
    service, vector_store, _, repository = _make_service(tmp_path, fake_client)
    _seed_workflow_chunk(vector_store, feature="Contact Log")

    generation_id = _ask_expecting_failure(service, ValidationError)

    assert repository.get_generation(generation_id).error.stage == KnowledgeAssistantErrorStage.RESPONSE_VALIDATION


def test_input_validation_failure_persists_an_error_record_even_though_nothing_else_was_persisted(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_VALID_ANSWER_NO_SOURCES)
    service, _, _, repository = _make_service(tmp_path, fake_client)

    with pytest.raises(ValidationError):
        service.ask(user_question="   ")
    [newest] = repository.list_generations()[:1]
    generation = repository.get_generation(newest.generation_id)

    assert generation.error is not None
    assert generation.error.stage == KnowledgeAssistantErrorStage.INPUT_VALIDATION
    assert generation.user_question is None
    assert generation.retrieved_chunks is None
    assert generation.prompt is None


def test_error_message_never_contains_an_openai_api_key(tmp_path):
    leaked_key_message = "request failed with header Authorization: Bearer sk-abcdefghij1234567890"
    fake_client = FakeOpenAIClient(dimension=2, chat_fail_with=RuntimeError(leaked_key_message))
    service, _, _, repository = _make_service(tmp_path, fake_client)

    generation_id = _ask_expecting_failure(service, ExternalServiceError)

    error_message = repository.get_generation(generation_id).error.error_message
    assert "sk-abcdefghij1234567890" not in error_message
    assert "[REDACTED]" in error_message


def test_a_failed_generation_directory_still_only_ever_has_one_generation_id(tmp_path):
    fake_client = FakeOpenAIClient(dimension=2, chat_fail_with=RuntimeError("connection reset"))
    service, _, _, repository = _make_service(tmp_path, fake_client)

    with pytest.raises(ExternalServiceError):
        _ask(service)

    assert len(repository.list_generations()) == 1


# --- Global retrieval: feature is provenance, never a restriction ---


def test_ask_retrieves_a_relevant_chunk_regardless_of_which_feature_indexed_it(tmp_path):
    """Feature is no longer a retrieval restriction — a chunk indexed
    under a feature never mentioned anywhere in the request must still
    be retrievable and citable.
    """
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_valid_answer(["Unrelated_Feature_Doc.pdf"]))
    service, vector_store, _, _ = _make_service(tmp_path, fake_client)
    _seed_chunk(
        vector_store,
        feature="Some Entirely Different Feature",
        chunk_id="cross-feature-chunk",
        source_filename="Unrelated_Feature_Doc.pdf",
        text="Bridged contacts are merged automatically when matching criteria align.",
    )

    _, answer = _ask(service)

    assert answer.source_documents == ["Unrelated_Feature_Doc.pdf"]


def test_ask_combines_relevant_chunks_from_multiple_features_in_one_answer(tmp_path):
    """A question can be answered from evidence in more than one
    feature folder at once — global retrieval must not force a single
    feature to be chosen.
    """
    fake_client = FakeOpenAIClient(
        dimension=2, chat_response_content=_valid_answer(["Feature_A_Doc.pdf", "Feature_B_Doc.pdf"])
    )
    service, vector_store, _, repository = _make_service(tmp_path, fake_client)
    _seed_chunk(
        vector_store,
        feature="Feature A",
        chunk_id="chunk-a",
        source_filename="Feature_A_Doc.pdf",
        text="Provider eligibility determines whether a provider can be assigned to a visit.",
    )
    _seed_chunk(
        vector_store,
        feature="Feature B",
        chunk_id="chunk-b",
        source_filename="Feature_B_Doc.pdf",
        text="Appointment eligibility determines whether a visit is billable.",
    )

    generation_id, answer = _ask(
        service, question="How does provider eligibility affect appointment eligibility?"
    )

    assert set(answer.source_documents) == {"Feature_A_Doc.pdf", "Feature_B_Doc.pdf"}
    documents_retrieved = repository.get_generation(generation_id).retrieved_chunks.documents_retrieved
    assert "Feature_A_Doc.pdf" in documents_retrieved
    assert "Feature_B_Doc.pdf" in documents_retrieved


def test_ask_persists_feature_as_provenance_on_each_retrieved_chunk(tmp_path):
    """Feature is dropped as a retrieval filter but must remain
    available as metadata/provenance on every retrieved chunk.
    """
    fake_client = FakeOpenAIClient(dimension=2, chat_response_content=_valid_answer(["Contact_Log_Workflow.pdf"]))
    service, vector_store, _, repository = _make_service(tmp_path, fake_client)
    _seed_workflow_chunk(vector_store, feature="Contact Log")

    generation_id, _ = _ask(service)

    [chunk] = repository.get_generation(generation_id).retrieved_chunks.chunks_retrieved
    assert chunk.feature == "Contact Log"


def test_ask_no_longer_accepts_or_requires_a_feature_argument(tmp_path):
    """`ask()` doesn't take `feature` at all any more — the Knowledge
    Assistant works purely from a question.
    """
    import inspect

    from app.services.knowledge_assistant_service import KnowledgeAssistantService

    parameters = inspect.signature(KnowledgeAssistantService.ask).parameters
    assert "feature" not in parameters
