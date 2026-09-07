import json
import re
import uuid

import chromadb
import pytest
import tiktoken
from chromadb.config import Settings as ChromaSettings

from app.core.exceptions import ExternalServiceError, NotFoundError
from app.retrievers.text_tokenizer import TOKENIZER_VERSION, tokenize
from app.services.bm25_index_manager import BM25IndexManager
from app.services.chunking_engine import ChunkingEngine
from app.services.embedding_service import EmbeddingService
from app.services.history_service import HistoryService
from app.services.index_service import IndexService
from app.services.parser_service import ParserService
from app.services.source_of_truth_indexer import SourceOfTruthIndexer
from app.services.upload_service import UploadService
from app.services.vector_store_service import VectorStoreService
from tests.fakes import FakeOpenAIClient

_MAX_UPLOAD_SIZE_BYTES = 1024 * 1024
_EMBEDDING_MODEL = "text-embedding-3-small"
_PRICE_PER_1K_TOKENS = 0.00002


def _write(root, feature: str, folder: str, filename: str, content: str) -> None:
    path = root / "source_of_truth" / feature / folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_service(
    tmp_path,
    fake_client: FakeOpenAIClient | None = None,
    price_per_1k_tokens: float = _PRICE_PER_1K_TOKENS,
) -> tuple[IndexService, VectorStoreService, FakeOpenAIClient]:
    # UploadService is only here to satisfy ParserService's constructor —
    # SourceOfTruthIndexer never uploads through it.
    unused_upload_service = UploadService(
        storage_root=tmp_path / "unused", max_upload_size_bytes=_MAX_UPLOAD_SIZE_BYTES
    )
    parser_service = ParserService(upload_service=unused_upload_service)
    indexer = SourceOfTruthIndexer(
        source_of_truth_root=tmp_path / "source_of_truth", parser_service=parser_service
    )
    chunking_engine = ChunkingEngine()

    client = fake_client if fake_client is not None else FakeOpenAIClient()
    embedding_service = EmbeddingService(client=client, model=_EMBEDDING_MODEL)

    # `chromadb.EphemeralClient()` instances share underlying storage for
    # identically-named collections within the same process, so each call
    # gets its own collection name to stay isolated across tests.
    chroma_client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)

    history_service = HistoryService(history_root=tmp_path / "history")
    bm25_index_manager = BM25IndexManager(
        index_dir=tmp_path / "bm25",
        collection_name=collection_name,
        tokenizer=tokenize,
        tokenizer_version=TOKENIZER_VERSION,
        chunking_signature="chunk_size=500,chunk_overlap=50",
    )

    service = IndexService(
        indexer=indexer,
        chunking_engine=chunking_engine,
        embedding_service=embedding_service,
        vector_store=vector_store,
        history_service=history_service,
        bm25_index_manager=bm25_index_manager,
        embedding_model=_EMBEDDING_MODEL,
        price_per_1k_tokens=price_per_1k_tokens,
    )
    return service, vector_store, client


def test_index_feature_raises_not_found_for_unknown_feature(tmp_path):
    service, _, _ = _make_service(tmp_path)

    with pytest.raises(NotFoundError):
        service.index_feature("Nonexistent")

    assert service.list_history() == []


def test_index_feature_indexes_workflow_test_case_and_issue_documents(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "onboarding.md", "# Onboarding\nSteps here that are long enough.")
    _write(tmp_path, "Appointments", "TestCases", "cases.csv", "id,title\n1,Book appointment\n")
    _write(tmp_path, "Appointments", "IssueSheets", "issues.txt", "KNOWN ISSUES\n\nSomething broke once.")
    service, vector_store, _ = _make_service(tmp_path)

    entry = service.index_feature("Appointments")

    assert entry.feature == "Appointments"
    assert entry.documents_indexed == 3
    assert entry.chunks_indexed >= 3
    stored = vector_store._collection.get(where={"feature": "Appointments"})
    artifact_types = {metadata["artifactType"] for metadata in stored["metadatas"]}
    assert artifact_types == {"WORKFLOW", "TEST_CASE", "ISSUE"}


def test_index_feature_stores_required_metadata_fields(tmp_path):
    content = "# Role Permissions\n\nUsers with CU role can edit records."
    _write(tmp_path, "Appointments", "workflows", "onboarding.md", content)
    service, vector_store, _ = _make_service(tmp_path)

    entry = service.index_feature("Appointments")

    stored = vector_store._collection.get(
        where={"feature": "Appointments"}, include=["embeddings", "documents", "metadatas"]
    )
    assert len(stored["ids"]) == entry.chunks_indexed == 1
    assert len(stored["embeddings"]) == 1
    assert stored["documents"][0] == "Role Permissions\n\nUsers with CU role can edit records."

    [metadata] = stored["metadatas"]
    assert metadata["chunkId"] == stored["ids"][0]
    assert metadata["artifactType"] == "WORKFLOW"
    assert metadata["feature"] == "Appointments"
    assert metadata["documentSource"] == "source_of_truth"
    assert metadata["sourceFilename"] == "onboarding.md"
    assert metadata["sectionHeading"] == "Role Permissions"
    assert metadata["parserName"] == "MarkdownParser"
    assert metadata["parserVersion"] == "1.0"
    assert "pageNumber" not in metadata  # Markdown has no pagination
    expected_tokens = len(tiktoken.encoding_for_model(_EMBEDDING_MODEL).encode(stored["documents"][0]))
    assert metadata["embeddingTokens"] == expected_tokens


def test_index_feature_persists_a_distinct_embedding_token_count_per_chunk(tmp_path):
    root = tmp_path / "source_of_truth" / "Appointments"
    (root / "workflows").mkdir(parents=True)
    (root / "workflows" / "short.txt").write_text("SHORT\n\nBrief.", encoding="utf-8")
    (root / "workflows" / "long.txt").write_text(
        "LONG\n\nThis section has quite a lot more words in its body than the other one does.",
        encoding="utf-8",
    )
    service, vector_store, _ = _make_service(tmp_path)

    service.index_feature("Appointments")

    stored = vector_store._collection.get(where={"feature": "Appointments"}, include=["documents", "metadatas"])
    by_filename = dict(zip((m["sourceFilename"] for m in stored["metadatas"]), stored["metadatas"], strict=True))
    assert by_filename["short.txt"]["embeddingTokens"] < by_filename["long.txt"]["embeddingTokens"]
    assert all(m["embeddingTokens"] > 0 for m in stored["metadatas"])


def test_index_feature_tracks_embedding_tokens_and_estimated_cost(tmp_path):
    content = "ROLE PERMISSIONS\n\nUsers with CU role can edit records today."
    _write(tmp_path, "Appointments", "workflows", "a.txt", content)
    fake_client = FakeOpenAIClient()
    service, _, _ = _make_service(tmp_path, fake_client=fake_client, price_per_1k_tokens=0.00002)

    entry = service.index_feature("Appointments")

    encoding = tiktoken.get_encoding("cl100k_base")
    expected_tokens = sum(len(encoding.encode(text)) for call in fake_client.calls for text in call)
    assert expected_tokens > 0
    assert entry.embedding_tokens == expected_tokens
    assert entry.estimated_embedding_cost == pytest.approx((expected_tokens / 1000) * 0.00002)
    assert entry.embedding_model == "text-embedding-3-small"
    assert entry.elapsed_seconds >= 0


def test_index_feature_with_no_chunkable_documents_skips_embedding_call(tmp_path):
    (tmp_path / "source_of_truth" / "EmptyFeature").mkdir(parents=True)
    fake_client = FakeOpenAIClient()
    service, vector_store, _ = _make_service(tmp_path, fake_client=fake_client)

    entry = service.index_feature("EmptyFeature")

    assert entry.documents_indexed == 0
    assert entry.chunks_indexed == 0
    assert entry.embedding_tokens == 0
    assert entry.estimated_embedding_cost == 0
    assert fake_client.calls == []
    stored = vector_store._collection.get(where={"feature": "EmptyFeature"})
    assert stored["ids"] == []


def test_index_feature_reindexing_replaces_rather_than_accumulates(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt", "ROLE PERMISSIONS\n\nSome body text.")
    service, vector_store, _ = _make_service(tmp_path)

    first_entry = service.index_feature("Appointments")
    second_entry = service.index_feature("Appointments")

    stored = vector_store._collection.get(where={"feature": "Appointments"})
    assert first_entry.chunks_indexed == second_entry.chunks_indexed
    assert len(stored["ids"]) == second_entry.chunks_indexed


# --- index_all_features (POST /index-source-of-truth) -------------------------


def test_index_all_features_indexes_every_discovered_feature(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.md", "# Onboarding\n\nSteps here that are long enough.")
    _write(tmp_path, "Coding Tool", "TestCases", "c.csv", "id,title\n1,Book an appointment\n")
    service, vector_store, _ = _make_service(tmp_path)

    summary = service.index_all_features()

    assert summary.total_features == 2
    assert summary.successful_features == 2
    assert summary.failed_features == 0
    # discovered in sorted order, one result per feature
    assert [result.feature for result in summary.features] == ["Appointments", "Coding Tool"]
    assert all(result.status == "SUCCESS" and result.error is None for result in summary.features)
    assert summary.total_documents_indexed == 2
    assert summary.total_chunks_indexed == sum(r.chunks_indexed for r in summary.features) >= 2
    assert summary.total_embedding_tokens == sum(r.embedding_tokens for r in summary.features) > 0

    # both features landed in ChromaDB and the persistent BM25 index
    stored = vector_store._collection.get(include=["metadatas"])
    assert {m["feature"] for m in stored["metadatas"]} == {"Appointments", "Coding Tool"}
    bm25_corpus = json.loads((tmp_path / "bm25" / "corpus.json").read_text(encoding="utf-8"))
    assert {r["feature"] for bucket in bm25_corpus.values() for r in bucket} == {"Appointments", "Coding Tool"}

    # history written once per feature, via the same path as index_feature
    assert [entry.feature for entry in service.list_history()] == ["Appointments", "Coding Tool"]


def test_index_all_features_runs_index_feature_once_per_feature_sequentially(tmp_path, monkeypatch):
    _write(tmp_path, "Appointments", "workflows", "a.md", "# Onboarding\n\nBody content long enough to chunk.")
    _write(tmp_path, "Coding Tool", "workflows", "b.md", "# Roles\n\nMore body content, also long enough.")
    service, _, _ = _make_service(tmp_path)

    calls: list[str] = []
    original = service.index_feature

    def spy(feature: str):
        calls.append(feature)
        return original(feature)

    monkeypatch.setattr(service, "index_feature", spy)

    service.index_all_features()

    assert calls == ["Appointments", "Coding Tool"]  # one call per feature, in sorted order


def test_index_all_features_reports_a_failed_feature_and_continues(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.md", "# Onboarding\n\nBody content long enough to chunk.")
    _write(tmp_path, "Broken", "workflows", "bad.pdf", "this is not a real pdf")
    _write(tmp_path, "Coding Tool", "workflows", "c.md", "# Roles\n\nMore body content, also long enough.")
    service, vector_store, _ = _make_service(tmp_path)

    summary = service.index_all_features()

    by_feature = {result.feature: result for result in summary.features}
    assert summary.total_features == 3
    assert summary.successful_features == 2
    assert summary.failed_features == 1
    assert by_feature["Broken"].status == "FAILED"
    assert by_feature["Broken"].error and "PDF" in by_feature["Broken"].error
    assert by_feature["Broken"].chunks_indexed == 0
    assert by_feature["Appointments"].status == "SUCCESS"
    assert by_feature["Coding Tool"].status == "SUCCESS"

    # only the two good features reached ChromaDB, BM25, and history
    stored = vector_store._collection.get(include=["metadatas"])
    assert {m["feature"] for m in stored["metadatas"]} == {"Appointments", "Coding Tool"}
    assert {entry.feature for entry in service.list_history()} == {"Appointments", "Coding Tool"}
    bm25_corpus = json.loads((tmp_path / "bm25" / "corpus.json").read_text(encoding="utf-8"))
    assert {r["feature"] for bucket in bm25_corpus.values() for r in bucket} == {"Appointments", "Coding Tool"}


def test_index_all_features_with_no_features_returns_an_empty_summary(tmp_path):
    (tmp_path / "source_of_truth").mkdir()
    service, _, _ = _make_service(tmp_path)

    summary = service.index_all_features()

    assert summary.total_features == 0
    assert summary.successful_features == 0
    assert summary.failed_features == 0
    assert summary.features == []
    assert summary.total_chunks_indexed == 0
    assert summary.total_embedding_tokens == 0
    assert service.list_history() == []


def test_index_feature_builds_and_persists_the_bm25_index_from_the_canonical_chunks(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.md", "# Encounter Status\n\nThe C_10 field drives transitions.")
    service, vector_store, _ = _make_service(tmp_path)

    service.index_feature("Appointments")

    corpus = json.loads((tmp_path / "bm25" / "corpus.json").read_text(encoding="utf-8"))
    records = corpus["WORKFLOW"]
    stored = vector_store._collection.get(where={"feature": "Appointments"}, include=["documents", "metadatas"])

    # BM25 corpus == the exact chunks that were embedded into ChromaDB
    assert {r["chunk_id"] for r in records} == set(stored["ids"])
    assert {r["text"] for r in records} == set(stored["documents"])
    assert records[0]["metadata"]["sectionHeading"] == "Encounter Status"
    assert records[0]["tokens"]  # tokenized at build time

    manifest = json.loads((tmp_path / "bm25" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunk_count"] == len(stored["ids"])
    assert manifest["features"] == {"Appointments": len(stored["ids"])}


def test_reindexing_a_feature_replaces_its_bm25_rows_without_touching_other_features(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.md", "# Alpha\n\nOriginal appointments workflow text.")
    _write(tmp_path, "Coding Tool", "workflows", "b.md", "# Beta\n\nCoding tool workflow text.")
    service, _, _ = _make_service(tmp_path)
    service.index_feature("Appointments")
    service.index_feature("Coding Tool")

    # change Appointments' content and re-index only it
    (tmp_path / "source_of_truth" / "Appointments" / "workflows" / "a.md").write_text(
        "# Alpha\n\nRevised appointments workflow text.", encoding="utf-8"
    )
    service.index_feature("Appointments")

    corpus = json.loads((tmp_path / "bm25" / "corpus.json").read_text(encoding="utf-8"))
    by_feature: dict[str, list[str]] = {}
    for record in corpus["WORKFLOW"]:
        by_feature.setdefault(record["feature"], []).append(record["text"])

    assert by_feature["Coding Tool"] == ["Beta\n\nCoding tool workflow text."]  # untouched
    assert by_feature["Appointments"] == ["Alpha\n\nRevised appointments workflow text."]  # replaced, not accumulated


def test_indexing_a_feature_with_no_chunks_clears_its_bm25_rows(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.md", "# Alpha\n\nSome text.")
    service, _, _ = _make_service(tmp_path)
    service.index_feature("Appointments")

    # remove the only document, re-index
    (tmp_path / "source_of_truth" / "Appointments" / "workflows" / "a.md").unlink()
    service.index_feature("Appointments")

    corpus = json.loads((tmp_path / "bm25" / "corpus.json").read_text(encoding="utf-8"))
    assert all(r["feature"] != "Appointments" for bucket in corpus.values() for r in bucket)


def test_bm25_index_is_queryable_immediately_after_indexing(tmp_path):
    from app.retrievers.bm25_retriever import BM25Retriever

    _write(tmp_path, "Appointments", "workflows", "a.md", "# Encounter Status\n\nThe C_10 field drives transitions.")
    _write(tmp_path, "Appointments", "TestCases", "t.csv", "id,summary\n1,Verify C_10 transition\n")
    service, _, _ = _make_service(tmp_path)
    service.index_feature("Appointments")

    retriever = BM25Retriever(service._bm25_index_manager)
    matches = retriever.retrieve("What does C_10 do?", top_k=10, filters={"artifactType": "WORKFLOW"})

    assert [m.chunk.section_heading for m in matches] == ["Encounter Status"]


def test_index_feature_creates_a_timestamped_history_folder_under_index_subfolder(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt", "ROLE PERMISSIONS\n\nSome body text.")
    service, vector_store, _ = _make_service(tmp_path)

    service.index_feature("Appointments")

    index_history_root = tmp_path / "history" / "index"
    [run_folder] = [p for p in index_history_root.iterdir() if p.is_dir()]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(_\d+)?", run_folder.name)

    summary = json.loads((run_folder / "index_summary.json").read_text(encoding="utf-8"))
    assert summary["feature"] == "Appointments"
    assert "indexed_at" in summary
    assert summary["average_tokens_per_chunk"] > 0
    assert summary["chroma_collection_name"] == vector_store.collection_name
    assert summary["index_status"] == "SUCCESS"

    preview = json.loads((run_folder / "embedding_preview.json").read_text(encoding="utf-8"))
    assert len(preview) == 1
    [chunk_preview] = preview
    assert chunk_preview["section_heading"] == "ROLE PERMISSIONS"
    assert chunk_preview["page_number"] is None
    assert chunk_preview["chunk_text"] == "ROLE PERMISSIONS\n\nSome body text."
    assert chunk_preview["embedding_tokens"] > 0
    assert chunk_preview["estimated_cost"] > 0
    assert "embedding" not in chunk_preview  # never the raw vector


def test_index_feature_does_not_create_history_when_embedding_generation_fails(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt", "ROLE PERMISSIONS\n\nSome body text.")
    failing_client = FakeOpenAIClient(fail_with=RuntimeError("simulated OpenAI outage"))
    service, vector_store, _ = _make_service(tmp_path, fake_client=failing_client)

    with pytest.raises(ExternalServiceError):
        service.index_feature("Appointments")

    assert service.list_history() == []
    assert not (tmp_path / "history").exists()
    stored = vector_store._collection.get(where={"feature": "Appointments"})
    assert stored["ids"] == []  # nothing was written to Chroma either


def test_index_feature_does_not_create_history_when_chroma_storage_fails(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt", "ROLE PERMISSIONS\n\nSome body text.")

    # Pre-seed the collection with a different embedding dimension so the
    # real add() call inside replace_feature_chunks fails for real — a
    # genuine ChromaDB error, not a mocked one.
    chroma_client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    seed_collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=None)
    seed_collection.add(
        ids=["seed"], embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]], documents=["seed"], metadatas=[{"feature": "Other"}]
    )
    vector_store = VectorStoreService(client=chroma_client, collection_name=collection_name)

    unused_upload_service = UploadService(
        storage_root=tmp_path / "unused", max_upload_size_bytes=_MAX_UPLOAD_SIZE_BYTES
    )
    parser_service = ParserService(upload_service=unused_upload_service)
    indexer = SourceOfTruthIndexer(source_of_truth_root=tmp_path / "source_of_truth", parser_service=parser_service)
    fake_client = FakeOpenAIClient(dimension=3)  # mismatches the seeded dimension of 5
    embedding_service = EmbeddingService(client=fake_client, model=_EMBEDDING_MODEL)
    history_service = HistoryService(history_root=tmp_path / "history")
    service = IndexService(
        indexer=indexer,
        chunking_engine=ChunkingEngine(),
        embedding_service=embedding_service,
        vector_store=vector_store,
        history_service=history_service,
        bm25_index_manager=BM25IndexManager(
            index_dir=tmp_path / "bm25",
            collection_name=collection_name,
            tokenizer=tokenize,
            tokenizer_version=TOKENIZER_VERSION,
            chunking_signature="chunk_size=500,chunk_overlap=50",
        ),
        embedding_model=_EMBEDDING_MODEL,
        price_per_1k_tokens=_PRICE_PER_1K_TOKENS,
    )

    with pytest.raises(ExternalServiceError):
        service.index_feature("Appointments")

    assert service.list_history() == []
    assert not (tmp_path / "history" / "index").exists()


def test_index_feature_stores_average_tokens_per_chunk_on_the_entry(tmp_path):
    root = tmp_path / "source_of_truth" / "Appointments" / "workflows"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("SHORT\n\nBrief.", encoding="utf-8")
    (root / "b.txt").write_text(
        "LONG\n\nThis section has quite a lot more words in its body than the other file does.",
        encoding="utf-8",
    )
    service, _, _ = _make_service(tmp_path)

    entry = service.index_feature("Appointments")

    assert entry.average_tokens_per_chunk == pytest.approx(entry.embedding_tokens / entry.chunks_indexed)


def test_list_history_returns_entries_in_the_order_they_were_indexed(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt", "ROLE PERMISSIONS\n\nBody one.")
    _write(tmp_path, "Coding Tool", "workflows", "b.txt", "ROLE PERMISSIONS\n\nBody two.")
    service, _, _ = _make_service(tmp_path)

    service.index_feature("Appointments")
    service.index_feature("Coding Tool")

    history = service.list_history()
    assert [entry.feature for entry in history] == ["Appointments", "Coding Tool"]


def test_get_latest_history_entry_raises_not_found_when_feature_never_indexed(tmp_path):
    service, _, _ = _make_service(tmp_path)

    with pytest.raises(NotFoundError):
        service.get_latest_history_entry("Nonexistent")


def test_get_latest_history_entry_returns_the_entry_for_that_feature(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt", "ROLE PERMISSIONS\n\nBody one.")
    service, _, _ = _make_service(tmp_path)
    indexed_entry = service.index_feature("Appointments")

    latest = service.get_latest_history_entry("Appointments")

    assert latest == indexed_entry


def test_get_latest_history_entry_ignores_entries_for_other_features(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt", "ROLE PERMISSIONS\n\nBody one.")
    _write(tmp_path, "Coding Tool", "workflows", "b.txt", "ROLE PERMISSIONS\n\nBody two.")
    service, _, _ = _make_service(tmp_path)
    service.index_feature("Appointments")
    coding_tool_entry = service.index_feature("Coding Tool")

    latest = service.get_latest_history_entry("Coding Tool")

    assert latest == coding_tool_entry


def test_get_latest_history_entry_returns_the_most_recent_run_after_reindexing(tmp_path):
    _write(tmp_path, "Appointments", "workflows", "a.txt", "ROLE PERMISSIONS\n\nBody one.")
    service, _, _ = _make_service(tmp_path)
    service.index_feature("Appointments")
    second_entry = service.index_feature("Appointments")

    latest = service.get_latest_history_entry("Appointments")

    assert latest.indexed_at == second_entry.indexed_at
