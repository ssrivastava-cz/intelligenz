import json
import re
import uuid

import chromadb
import pytest
import tiktoken
from chromadb.config import Settings as ChromaSettings

from app.core.exceptions import ExternalServiceError, NotFoundError
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

    service = IndexService(
        indexer=indexer,
        chunking_engine=chunking_engine,
        embedding_service=embedding_service,
        vector_store=vector_store,
        history_service=history_service,
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
