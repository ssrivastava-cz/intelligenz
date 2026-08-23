"""Unit tests for UploadEmbeddingService — the uploaded-document
equivalent of IndexService, but embedding into a dedicated per-session
ChromaDB collection instead of the Source of Truth collection.
"""
import io
import json

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings
from fastapi import UploadFile

from app.core.exceptions import ExternalServiceError, NotFoundError
from app.services.chunking_engine import ChunkingEngine
from app.services.embedding_service import EmbeddingService
from app.services.history_service import HistoryService
from app.services.parser_service import ParserService
from app.services.upload_embedding_service import UploadEmbeddingService
from app.services.upload_service import UploadService
from app.services.uploaded_document_indexer import UploadedDocumentIndexer
from tests.fakes import FakeOpenAIClient

_MAX_UPLOAD_SIZE_BYTES = 1024 * 1024
_EMBEDDING_MODEL = "text-embedding-3-small"
_PRICE_PER_1K_TOKENS = 0.00002
_COLLECTION_PREFIX = "uploaded_documents"


def _upload_file(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


def _make_service(tmp_path, fake_client: FakeOpenAIClient | None = None):
    upload_service = UploadService(
        storage_root=tmp_path / "uploads_root", max_upload_size_bytes=_MAX_UPLOAD_SIZE_BYTES
    )
    parser_service = ParserService(upload_service=upload_service)
    upload_indexer = UploadedDocumentIndexer(upload_service=upload_service, parser_service=parser_service)

    client = fake_client if fake_client is not None else FakeOpenAIClient()
    embedding_service = EmbeddingService(client=client, model=_EMBEDDING_MODEL)

    chroma_client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    history_service = HistoryService(history_root=tmp_path / "history")

    service = UploadEmbeddingService(
        indexer=upload_indexer,
        chunking_engine=ChunkingEngine(),
        embedding_service=embedding_service,
        chroma_client=chroma_client,
        history_service=history_service,
        embedding_model=_EMBEDDING_MODEL,
        price_per_1k_tokens=_PRICE_PER_1K_TOKENS,
        collection_prefix=_COLLECTION_PREFIX,
    )
    return service, upload_service, chroma_client, client


async def _make_session_with_document(
    upload_service, filename: str = "a.md", content: bytes = b"# Role Permissions\n\nBody text."
) -> str:
    session = upload_service.create_session()
    await upload_service.upload_user_documents(
        session_id=session.id, feature=session.id, files=[_upload_file(filename, content)]
    )
    return session.id


async def test_embed_session_raises_not_found_for_unknown_session(tmp_path):
    service, _, _, _ = _make_service(tmp_path)

    with pytest.raises(NotFoundError):
        service.embed_session("nonexistent-session")


async def test_embed_session_with_no_documents_produces_zero_chunk_history(tmp_path):
    service, upload_service, _, fake_client = _make_service(tmp_path)
    session = upload_service.create_session()

    entry = service.embed_session(session.id)

    assert entry.documents_indexed == 0
    assert entry.chunks_indexed == 0
    assert entry.embedding_tokens == 0
    assert fake_client.calls == []


async def test_embed_session_stores_chunks_in_a_dedicated_collection(tmp_path):
    service, upload_service, chroma_client, _ = _make_service(tmp_path)
    session_id = await _make_session_with_document(upload_service)

    entry = service.embed_session(session_id)

    assert entry.chroma_collection_name == f"{_COLLECTION_PREFIX}_{session_id}"
    collection = chroma_client.get_or_create_collection(
        name=entry.chroma_collection_name, embedding_function=None
    )
    stored = collection.get(where={"feature": session_id})
    assert len(stored["ids"]) == entry.chunks_indexed == 1


async def test_embed_session_persists_history_under_history_upload(tmp_path):
    service, upload_service, _, _ = _make_service(tmp_path)
    session_id = await _make_session_with_document(upload_service)

    service.embed_session(session_id)

    upload_history_root = tmp_path / "history" / "upload"
    [run_folder] = [p for p in upload_history_root.iterdir() if p.is_dir()]
    summary = json.loads((run_folder / "uploaded_embedding_summary.json").read_text(encoding="utf-8"))
    assert summary["upload_session_id"] == session_id
    assert summary["index_status"] == "SUCCESS"
    preview = json.loads((run_folder / "embedding_preview.json").read_text(encoding="utf-8"))
    assert len(preview) == 1


async def test_embed_session_reembedding_replaces_rather_than_accumulates(tmp_path):
    service, upload_service, chroma_client, _ = _make_service(tmp_path)
    session_id = await _make_session_with_document(upload_service)

    first_entry = service.embed_session(session_id)
    second_entry = service.embed_session(session_id)

    collection = chroma_client.get_or_create_collection(
        name=second_entry.chroma_collection_name, embedding_function=None
    )
    stored = collection.get(where={"feature": session_id})
    assert len(stored["ids"]) == second_entry.chunks_indexed == first_entry.chunks_indexed


async def test_embed_session_does_not_affect_a_different_sessions_collection(tmp_path):
    service, upload_service, chroma_client, _ = _make_service(tmp_path)
    session_a = await _make_session_with_document(upload_service, filename="a.md", content=b"# A\n\nBody A.")
    session_b = await _make_session_with_document(upload_service, filename="b.md", content=b"# B\n\nBody B.")

    entry_a = service.embed_session(session_a)
    entry_b = service.embed_session(session_b)

    assert entry_a.chroma_collection_name != entry_b.chroma_collection_name
    collection_a = chroma_client.get_or_create_collection(
        name=entry_a.chroma_collection_name, embedding_function=None
    )
    assert collection_a.get(where={"feature": session_a})["ids"]
    assert collection_a.get(where={"feature": session_b})["ids"] == []


async def test_get_latest_history_raises_not_found_when_never_embedded(tmp_path):
    service, _, _, _ = _make_service(tmp_path)

    with pytest.raises(NotFoundError):
        service.get_latest_history("nonexistent-session")


async def test_get_latest_history_returns_the_record_for_that_session(tmp_path):
    service, upload_service, _, _ = _make_service(tmp_path)
    session_id = await _make_session_with_document(upload_service)
    entry = service.embed_session(session_id)

    record = service.get_latest_history(session_id)

    assert record.summary == entry
    assert len(record.embedding_preview) == 1


async def test_get_latest_history_returns_the_most_recent_run_after_reembedding(tmp_path):
    service, upload_service, _, _ = _make_service(tmp_path)
    session_id = await _make_session_with_document(upload_service)
    service.embed_session(session_id)
    second_entry = service.embed_session(session_id)

    record = service.get_latest_history(session_id)

    assert record.summary.uploaded_at == second_entry.uploaded_at


async def test_embed_session_does_not_create_history_when_embedding_fails(tmp_path):
    failing_client = FakeOpenAIClient(fail_with=RuntimeError("simulated outage"))
    service, upload_service, _, _ = _make_service(tmp_path, fake_client=failing_client)
    session_id = await _make_session_with_document(upload_service)

    with pytest.raises(ExternalServiceError):
        service.embed_session(session_id)

    assert not (tmp_path / "history").exists()
