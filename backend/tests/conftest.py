"""Shared pytest fixtures."""
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_bm25_index_manager,
    get_parser_service,
    get_source_of_truth_indexer,
    get_test_plan_service,
    get_upload_service,
)
from app.main import app
from app.retrievers.text_tokenizer import TOKENIZER_VERSION, tokenize
from app.services.bm25_index_manager import BM25IndexManager
from app.services.source_of_truth_indexer import SourceOfTruthIndexer
from app.services.upload_service import UploadService
from tests.directory_snapshot import diff_snapshots, snapshot_directory

_TEST_MAX_UPLOAD_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB — small so oversized-file tests stay fast

# The application's real runtime data locations — everything under
# backend/data/ (history/, redmine/, chroma/), plus source_of_truth/,
# uploads/, and database/ (the Knowledge Assistant's filesystem
# GenerationRepository — real questions asked against a running server
# populate it, exactly like backend/data/history/ does for Test Plan
# Generation), all of which live directly under backend/ rather than
# under data/. Every test must isolate itself to tmp_path (or an
# in-memory double) instead of resolving any of these for real;
# _protect_real_application_data below enforces that for the whole
# suite, regardless of which tests run or in what order.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROTECTED_REAL_DIRECTORIES = [
    _BACKEND_DIR / "data",
    _BACKEND_DIR / "source_of_truth",
    _BACKEND_DIR / "uploads",
    _BACKEND_DIR / "database",
]


@pytest.fixture(scope="session", autouse=True)
def _protect_real_application_data():
    """Session-wide regression guard: fails the test run if anything
    under the application's real runtime data directories is created,
    modified, or deleted while the suite runs. This is the test-isolation
    contract every fixture and test in this suite must uphold — see
    `tests/unit/test_directory_snapshot.py` for proof the underlying
    snapshot mechanism itself correctly detects such changes.
    """
    before = {root: snapshot_directory(root) for root in _PROTECTED_REAL_DIRECTORIES}
    yield
    after = {root: snapshot_directory(root) for root in _PROTECTED_REAL_DIRECTORIES}

    failures = [
        f"{root}: {diff_snapshots(before[root], after[root])}"
        for root in _PROTECTED_REAL_DIRECTORIES
        if before[root] != after[root]
    ]
    assert not failures, (
        "Test suite modified real application data under backend/ — every test "
        "must isolate itself to tmp_path instead:\n" + "\n".join(failures)
    )


@pytest.fixture
def upload_service(tmp_path):
    """Isolates document storage to a per-test temp directory instead of
    writing into the real backend/data/ folder.
    """
    service = UploadService(storage_root=tmp_path, max_upload_size_bytes=_TEST_MAX_UPLOAD_SIZE_BYTES)
    app.dependency_overrides[get_upload_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_upload_service, None)


@pytest.fixture
def test_plan_service(upload_service):
    """The exact TestPlanService instance the app resolves for requests
    during this test (same `upload_service` argument -> same lru_cache
    entry), for direct assertions on generation/session state.
    """
    return get_test_plan_service(upload_service=upload_service)


@pytest.fixture
def source_of_truth_indexer(upload_service, tmp_path):
    """Points the indexer at the same isolated tmp_path as `upload_service`
    (rather than the real STORAGE_ROOT from settings), so integration
    tests can set up a `source_of_truth/` tree under `tmp_path` directly.
    """
    parser_service = get_parser_service(upload_service=upload_service)
    indexer = SourceOfTruthIndexer(
        source_of_truth_root=tmp_path / "source_of_truth",
        parser_service=parser_service,
    )
    app.dependency_overrides[get_source_of_truth_indexer] = lambda: indexer
    yield indexer
    app.dependency_overrides.pop(get_source_of_truth_indexer, None)


@pytest.fixture
def bm25_index_manager(tmp_path):
    """Isolates the persistent BM25 index to a per-test temp directory so
    indexing endpoints never write into the real backend/data/bm25/."""
    manager = BM25IndexManager(
        index_dir=tmp_path / "bm25",
        collection_name="source_of_truth_chunks",
        tokenizer=tokenize,
        tokenizer_version=TOKENIZER_VERSION,
        chunking_signature="chunk_size=500,chunk_overlap=50",
    )
    app.dependency_overrides[get_bm25_index_manager] = lambda: manager
    yield manager
    app.dependency_overrides.pop(get_bm25_index_manager, None)


@pytest.fixture
async def client(upload_service, source_of_truth_indexer, bm25_index_manager):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
