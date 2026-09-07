"""Regression tests proving `backend/data/history/` and
`backend/data/redmine/` are fully independent siblings under `data/`:
creating, writing to, or reading from one must never create, modify, or
delete anything under the other.

Written after a reported regression where `backend/data/history/`
disappeared following Redmine persistence work. Investigation (see the
accompanying report) found no code path in `HistoryService`,
`RedmineService`, `UploadService`, `IndexService`, dependency
injection, or application startup that touches the other subsystem's
directory — both services only ever operate under their own
independently-configured root, exactly mirroring how `settings.py`
gives them distinct `history_root`/`redmine_data_root` values that
happen to share a `data/` parent purely by configuration convention,
not by any shared code path. These tests lock that independence in so
a future change can't quietly reintroduce cross-subsystem interference.
"""
import chromadb
import httpx
from chromadb.config import Settings as ChromaSettings

from app.retrievers.text_tokenizer import TOKENIZER_VERSION, tokenize
from app.services.bm25_index_manager import BM25IndexManager
from app.services.chunking_engine import ChunkingEngine
from app.services.embedding_service import EmbeddingService
from app.services.history_service import HistoryService
from app.services.index_service import IndexService
from app.services.parser_service import ParserService
from app.services.redmine_service import RedmineService
from app.services.source_of_truth_indexer import SourceOfTruthIndexer
from app.services.upload_service import UploadService
from app.services.vector_store_service import VectorStoreService
from tests.fakes import FakeOpenAIClient

_REDMINE_BASE_URL = "https://redmine.example.com"


def _write_source_document(tmp_path, feature: str) -> None:
    root = tmp_path / "source_of_truth" / feature / "workflows"
    root.mkdir(parents=True)
    (root / "onboarding.md").write_text("# Role Permissions\n\nBody text.", encoding="utf-8")


def _make_index_service(tmp_path, history_root) -> IndexService:
    unused_upload_service = UploadService(
        storage_root=tmp_path / "unused", max_upload_size_bytes=1024 * 1024
    )
    parser_service = ParserService(upload_service=unused_upload_service)
    indexer = SourceOfTruthIndexer(
        source_of_truth_root=tmp_path / "source_of_truth", parser_service=parser_service
    )
    chroma_client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    vector_store = VectorStoreService(client=chroma_client, collection_name="test_collection")

    return IndexService(
        indexer=indexer,
        chunking_engine=ChunkingEngine(),
        embedding_service=EmbeddingService(client=FakeOpenAIClient(), model="text-embedding-3-small"),
        vector_store=vector_store,
        history_service=HistoryService(history_root=history_root),
        bm25_index_manager=BM25IndexManager(
            index_dir=tmp_path / "bm25",
            collection_name="test_collection",
            tokenizer=tokenize,
            tokenizer_version=TOKENIZER_VERSION,
            chunking_signature="chunk_size=500,chunk_overlap=50",
        ),
        embedding_model="text-embedding-3-small",
        price_per_1k_tokens=0.00002,
    )


def _issue_payload(ticket_id: str = "12345") -> dict:
    return {
        "id": int(ticket_id),
        "project": {"id": 1, "name": "Cozeva Platform"},
        "status": {"id": 1, "name": "New"},
        "author": {"id": 1, "name": "Jane Doe"},
        "subject": "Fix contact tracking bug",
        "description": "Steps to reproduce...",
        "created_on": "2026-08-01T10:00:00Z",
        "updated_on": "2026-08-02T12:00:00Z",
        "custom_fields": [],
        "attachments": [],
    }


def _make_redmine_service(redmine_data_root) -> RedmineService:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issue": _issue_payload()})

    client = httpx.Client(
        base_url=_REDMINE_BASE_URL,
        headers={"X-Redmine-API-Key": "test-key"},
        transport=httpx.MockTransport(handler),
    )
    return RedmineService(client=client, data_root=redmine_data_root)


def _snapshot(path) -> dict[str, bytes]:
    """Every file's relative path and exact bytes under `path` — for
    asserting a directory tree is byte-for-byte unchanged before/after
    some unrelated action, not just "still exists."
    """
    if not path.is_dir():
        return {}
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_indexing_creates_history_under_data_history_index(tmp_path):
    data_root = tmp_path / "data"
    _write_source_document(tmp_path, "Appointments")
    service = _make_index_service(tmp_path, history_root=data_root / "history")

    service.index_feature("Appointments")

    index_root = data_root / "history" / "index"
    assert index_root.is_dir()
    [run_folder] = [p for p in index_root.iterdir() if p.is_dir()]
    assert (run_folder / "index_summary.json").is_file()
    assert (run_folder / "embedding_preview.json").is_file()


def test_redmine_persistence_creates_only_redmine_data(tmp_path):
    data_root = tmp_path / "data"
    redmine_service = _make_redmine_service(data_root / "redmine")

    redmine_service.get_ticket("12345")

    assert (data_root / "redmine" / "12345").is_dir()
    assert not (data_root / "history").exists()  # indexing never ran; nothing should have created it


def test_indexing_then_redmine_preserves_history(tmp_path):
    data_root = tmp_path / "data"
    _write_source_document(tmp_path, "Appointments")
    index_service = _make_index_service(tmp_path, history_root=data_root / "history")
    redmine_service = _make_redmine_service(data_root / "redmine")

    index_service.index_feature("Appointments")
    history_before = _snapshot(data_root / "history")
    assert history_before  # sanity: something was actually written

    redmine_service.get_ticket("12345")

    assert _snapshot(data_root / "history") == history_before
    assert (data_root / "redmine" / "12345").is_dir()


def test_redmine_then_indexing_preserves_redmine_data(tmp_path):
    data_root = tmp_path / "data"
    _write_source_document(tmp_path, "Appointments")
    index_service = _make_index_service(tmp_path, history_root=data_root / "history")
    redmine_service = _make_redmine_service(data_root / "redmine")

    redmine_service.get_ticket("12345")
    redmine_before = _snapshot(data_root / "redmine")
    assert redmine_before

    index_service.index_feature("Appointments")

    assert _snapshot(data_root / "redmine") == redmine_before
    assert (data_root / "history" / "index").is_dir()


def test_history_and_redmine_directories_coexist_as_siblings(tmp_path):
    data_root = tmp_path / "data"
    _write_source_document(tmp_path, "Appointments")
    index_service = _make_index_service(tmp_path, history_root=data_root / "history")
    redmine_service = _make_redmine_service(data_root / "redmine")

    index_service.index_feature("Appointments")
    redmine_service.get_ticket("12345")

    assert (data_root / "history").is_dir()
    assert (data_root / "redmine").is_dir()
    assert {"history", "redmine"} <= {p.name for p in data_root.iterdir()}


def test_repeated_indexing_creates_new_folders_without_removing_previous_runs(tmp_path):
    data_root = tmp_path / "data"
    _write_source_document(tmp_path, "Appointments")
    service = _make_index_service(tmp_path, history_root=data_root / "history")

    service.index_feature("Appointments")
    first_run_folders = {p.name for p in (data_root / "history" / "index").iterdir() if p.is_dir()}

    service.index_feature("Appointments")
    second_run_folders = {p.name for p in (data_root / "history" / "index").iterdir() if p.is_dir()}

    assert first_run_folders.issubset(second_run_folders)
    assert len(second_run_folders) > len(first_run_folders)


def test_redmine_never_deletes_a_pre_existing_history_directory(tmp_path):
    """Simulates the reported regression directly: a history/ directory
    already exists (from a prior indexing run) before RedmineService
    does anything — it must still be there, byte-for-byte, afterward.
    """
    data_root = tmp_path / "data"
    history_dir = data_root / "history" / "index" / "2026-08-01_00-00-00"
    history_dir.mkdir(parents=True)
    (history_dir / "index_summary.json").write_text('{"feature": "Appointments"}', encoding="utf-8")
    (history_dir / "embedding_preview.json").write_text("[]", encoding="utf-8")
    history_before = _snapshot(data_root / "history")

    redmine_service = _make_redmine_service(data_root / "redmine")
    redmine_service.get_ticket("12345")

    assert _snapshot(data_root / "history") == history_before


def test_indexing_never_deletes_a_pre_existing_redmine_directory(tmp_path):
    data_root = tmp_path / "data"
    redmine_dir = data_root / "redmine" / "99999"
    redmine_dir.mkdir(parents=True)
    (redmine_dir / "ticket.json").write_text('{"id": 99999}', encoding="utf-8")
    redmine_before = _snapshot(data_root / "redmine")

    _write_source_document(tmp_path, "Appointments")
    index_service = _make_index_service(tmp_path, history_root=data_root / "history")
    index_service.index_feature("Appointments")

    assert _snapshot(data_root / "redmine") == redmine_before
