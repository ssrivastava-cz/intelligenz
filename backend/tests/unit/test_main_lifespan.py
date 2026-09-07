"""The FastAPI startup hook loads the persistent BM25 index once — it
must not rebuild it just because the process restarted, and must start
cleanly (with a clear log) when no index has been persisted yet.
"""
import logging
from pathlib import Path

import pytest

from app import main
from app.models.bm25_index import BM25ChunkRecord
from app.models.common import DocumentCategory
from app.retrievers.text_tokenizer import TOKENIZER_VERSION, tokenize
from app.services.bm25_index_manager import BM25IndexManager


def _manager(index_dir: Path) -> BM25IndexManager:
    return BM25IndexManager(
        index_dir=index_dir,
        collection_name="source_of_truth_chunks",
        tokenizer=tokenize,
        tokenizer_version=TOKENIZER_VERSION,
        chunking_signature="chunk_size=500,chunk_overlap=50",
    )


@pytest.mark.asyncio
async def test_lifespan_loads_an_existing_persistent_index(tmp_path, monkeypatch):
    text = "Provider eligibility workflow."
    builder = _manager(tmp_path / "bm25")
    builder.rebuild_all([BM25ChunkRecord.from_chunk("c1", text, _meta("c1"), tokenize(text))])
    manager = _manager(tmp_path / "bm25")
    monkeypatch.setattr(main, "get_bm25_index_manager", lambda: manager)

    async with main.lifespan(main.app):
        assert manager.is_loaded() is True

    hits = manager.search(DocumentCategory.WORKFLOW, tokenize("provider eligibility"), top_k=5)
    assert [h.record.chunk_id for h in hits] == ["c1"]


@pytest.mark.asyncio
async def test_lifespan_starts_cleanly_and_warns_when_no_index_exists(tmp_path, monkeypatch, caplog):
    manager = _manager(tmp_path / "missing")
    monkeypatch.setattr(main, "get_bm25_index_manager", lambda: manager)

    with caplog.at_level(logging.WARNING):
        async with main.lifespan(main.app):
            pass

    assert manager.is_loaded() is False
    assert any("No persistent BM25 index" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_lifespan_never_rebuilds_the_index(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "missing")

    def _boom(*_a, **_k):
        raise AssertionError("startup must not rebuild the BM25 index from source")

    monkeypatch.setattr(manager, "rebuild_feature", _boom)
    monkeypatch.setattr(manager, "rebuild_all", _boom)
    monkeypatch.setattr(main, "get_bm25_index_manager", lambda: manager)

    async with main.lifespan(main.app):
        pass


def _meta(chunk_id: str) -> dict:
    return {
        "chunkId": chunk_id,
        "artifactType": "WORKFLOW",
        "feature": "Service Model 1",
        "documentSource": "source_of_truth",
        "sourceFilename": "doc.md",
        "parserName": "MarkdownParser",
        "parserVersion": "1.0",
    }
