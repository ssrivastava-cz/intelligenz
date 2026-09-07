"""Unit tests for `BM25IndexManager` — the build/save/load/rebuild
lifecycle of the persistent global lexical index. No ChromaDB, no
OpenAI: the manager operates purely on `BM25ChunkRecord`s (the canonical
chunks) and its own JSON files.
"""
import json

import pytest
from rank_bm25 import BM25Okapi

from app.models.bm25_index import BM25ChunkRecord
from app.models.common import DocumentCategory
from app.retrievers.text_tokenizer import TOKENIZER_VERSION, tokenize
from app.services.bm25_index_manager import BM25IndexManager

_COLLECTION = "source_of_truth_chunks"


def _manager(tmp_path, subdir="bm25") -> BM25IndexManager:
    return BM25IndexManager(
        index_dir=tmp_path / subdir,
        collection_name=_COLLECTION,
        tokenizer=tokenize,
        tokenizer_version=TOKENIZER_VERSION,
        chunking_signature="chunk_size=500,chunk_overlap=50",
    )


def _record(chunk_id, text, *, feature="Service Model 1", artifact_type="WORKFLOW", **metadata) -> BM25ChunkRecord:
    meta = {
        "chunkId": chunk_id,
        "artifactType": artifact_type,
        "feature": feature,
        "documentSource": "source_of_truth",
        "sourceFilename": "doc.md",
        "parserName": "MarkdownParser",
        "parserVersion": "1.0",
        **metadata,
    }
    return BM25ChunkRecord.from_chunk(chunk_id, text, meta, tokenize(text))


def test_build_from_canonical_chunks_and_query(tmp_path):
    manager = _manager(tmp_path)
    manager.rebuild_feature(
        "Service Model 1",
        [
            _record("c1", "Provider participation gates appointment eligibility for a visit."),
            _record("c2", "General onboarding notes for new platform users."),
        ],
    )

    hits = manager.search(DocumentCategory.WORKFLOW, tokenize("provider participation eligibility"), top_k=10)

    assert [h.record.chunk_id for h in hits] == ["c1"]


def test_save_writes_corpus_and_manifest(tmp_path):
    manager = _manager(tmp_path)
    manager.rebuild_feature("Service Model 1", [_record("c1", "Provider eligibility rules for a visit.")])

    corpus = json.loads((tmp_path / "bm25" / "corpus.json").read_text())
    manifest = json.loads((tmp_path / "bm25" / "manifest.json").read_text())

    assert [r["chunk_id"] for r in corpus["WORKFLOW"]] == ["c1"]
    assert corpus["WORKFLOW"][0]["tokens"] == tokenize("Provider eligibility rules for a visit.")
    assert manifest["chunk_count"] == 1
    assert manifest["tokenizer_version"] == TOKENIZER_VERSION
    assert manifest["chunking_signature"] == "chunk_size=500,chunk_overlap=50"
    assert manifest["features"] == {"Service Model 1": 1}
    assert manifest["bm25_params"]["variant"] == "BM25Okapi"
    assert manifest["corpus_sha256"]


def test_persisted_index_loads_in_a_fresh_manager_after_restart(tmp_path):
    builder = _manager(tmp_path)
    builder.rebuild_feature(
        "Service Model 1",
        [
            _record("c1", "Provider participation gates appointment eligibility."),
            _record("c2", "Enrollment must cover the visit date of service."),
        ],
    )

    reloaded = _manager(tmp_path)  # new process / new instance, same dir
    assert reloaded.is_available()
    assert not reloaded.is_loaded()
    reloaded.load()
    assert reloaded.is_loaded()

    hits = reloaded.search(DocumentCategory.WORKFLOW, tokenize("enrollment visit date"), top_k=10)
    assert [h.record.chunk_id for h in hits] == ["c2"]


def test_loaded_index_ranks_identically_to_an_in_memory_build(tmp_path):
    """The important validation: for a fixed corpus and fixed BM25
    parameters, persisting then loading must not change rankings vs an
    index built in memory immediately before querying."""
    records = [
        _record("c1", "Delete and undo delete both apply directly to a Contact Log entry."),
        _record("c2", "Contact Log is one of many features available in the product."),
        _record("c3", "Bridged contacts are merged automatically when matching criteria align."),
        _record("c4", "Provider participation gates appointment eligibility for a visit."),
    ]
    query = tokenize("How do I delete and undo delete a Contact Log entry?")

    # (1) in-memory, exactly like the old rebuild-per-query path
    in_memory = BM25Okapi([r.tokens for r in records])
    scores = in_memory.get_scores(query)
    qset = set(query)
    expected = [
        r.chunk_id
        for r, _ in sorted(
            ((r, s) for r, s in zip(records, scores, strict=True) if qset & set(r.tokens)),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    # (2) persisted then loaded
    _manager(tmp_path).rebuild_feature("Service Model 1", records)
    loaded = _manager(tmp_path)
    loaded.load()
    hits = loaded.search(DocumentCategory.WORKFLOW, query, top_k=10)

    assert [h.record.chunk_id for h in hits] == expected
    assert expected  # sanity: the query actually matched something


def test_rebuild_feature_replaces_only_that_feature(tmp_path):
    manager = _manager(tmp_path)
    manager.rebuild_feature("Feature A", [_record("a1", "Alpha eligibility workflow.", feature="Feature A")])
    manager.rebuild_feature("Feature B", [_record("b1", "Beta eligibility workflow.", feature="Feature B")])

    # re-index Feature A with a different chunk set
    manager.rebuild_feature("Feature A", [_record("a2", "Alpha eligibility workflow, revised.", feature="Feature A")])

    corpus = json.loads((tmp_path / "bm25" / "corpus.json").read_text())
    workflow_ids = {r["chunk_id"] for r in corpus["WORKFLOW"]}
    assert workflow_ids == {"a2", "b1"}  # a1 gone, b1 untouched
    assert manager.manifest().features == {"Feature A": 1, "Feature B": 1}


def test_empty_records_drops_a_feature(tmp_path):
    manager = _manager(tmp_path)
    manager.rebuild_feature("Feature A", [_record("a1", "Alpha workflow.", feature="Feature A")])
    manager.rebuild_feature("Feature B", [_record("b1", "Beta workflow.", feature="Feature B")])

    manager.rebuild_feature("Feature A", [])

    corpus = json.loads((tmp_path / "bm25" / "corpus.json").read_text())
    assert {r["chunk_id"] for r in corpus["WORKFLOW"]} == {"b1"}


def test_metadata_and_chunk_id_survive_the_round_trip(tmp_path):
    _manager(tmp_path).rebuild_feature(
        "Service Model 1",
        [
            _record(
                "c1",
                "Provider eligibility for a visit.",
                artifact_type="TEST_CASE",
                sectionHeading="Provider Eligibility",
                pageNumber=3,
                documentId="doc-1",
                chunkNumber=7,
                documentTitle="RA Eligibility BRD",
                sourcePath="source_of_truth/Service Model 1/TestCases/cases.pdf",
                sourceFolder="Service Model 1",
            )
        ],
    )
    loaded = _manager(tmp_path)
    loaded.load()

    [hit] = loaded.search(DocumentCategory.TEST_CASE, tokenize("provider eligibility visit"), top_k=10)
    record = hit.record
    assert record.chunk_id == "c1"
    assert record.feature == "Service Model 1"
    assert record.artifact_type == DocumentCategory.TEST_CASE
    assert record.metadata["sectionHeading"] == "Provider Eligibility"
    assert record.metadata["pageNumber"] == 3
    assert record.metadata["documentId"] == "doc-1"
    assert record.metadata["chunkNumber"] == 7
    assert record.metadata["documentTitle"] == "RA Eligibility BRD"
    assert record.metadata["sourcePath"] == "source_of_truth/Service Model 1/TestCases/cases.pdf"
    assert record.metadata["sourceFolder"] == "Service Model 1"


def test_missing_index_is_handled_gracefully(tmp_path):
    manager = _manager(tmp_path, subdir="does-not-exist")

    assert manager.is_available() is False
    assert manager.is_loaded() is False
    assert manager.ensure_loaded() is False
    assert manager.search(DocumentCategory.WORKFLOW, tokenize("anything"), top_k=5) == []
    with pytest.raises(FileNotFoundError):
        manager.load()


def test_search_is_scoped_to_its_artifact_type_bucket(tmp_path):
    manager = _manager(tmp_path)
    manager.rebuild_feature(
        "Service Model 1",
        [
            _record("w1", "Eligibility workflow description.", artifact_type="WORKFLOW"),
            _record("t1", "Eligibility test case steps.", artifact_type="TEST_CASE"),
            _record("i1", "Eligibility issue report.", artifact_type="ISSUE"),
        ],
    )

    workflow_hits = manager.search(DocumentCategory.WORKFLOW, tokenize("eligibility"), top_k=10)
    assert [h.record.chunk_id for h in workflow_hits] == ["w1"]


def test_feature_post_filter_restricts_results_without_changing_idf(tmp_path):
    manager = _manager(tmp_path)
    manager.rebuild_feature("Feature A", [_record("a1", "Provider eligibility rules.", feature="Feature A")])
    manager.rebuild_feature("Feature B", [_record("b1", "Provider eligibility rules.", feature="Feature B")])

    unscoped = manager.search(DocumentCategory.WORKFLOW, tokenize("provider eligibility"), top_k=10)
    scoped = manager.search(DocumentCategory.WORKFLOW, tokenize("provider eligibility"), top_k=10, feature="Feature A")

    assert {h.record.chunk_id for h in unscoped} == {"a1", "b1"}
    assert [h.record.chunk_id for h in scoped] == ["a1"]


def test_is_stale_detects_a_tokenizer_version_change(tmp_path):
    _manager(tmp_path).rebuild_feature("Service Model 1", [_record("c1", "Eligibility workflow.")])

    same = _manager(tmp_path)
    same.load()
    assert same.is_stale() is False

    drifted = BM25IndexManager(
        index_dir=tmp_path / "bm25",
        collection_name=_COLLECTION,
        tokenizer=tokenize,
        tokenizer_version="999",
        chunking_signature="chunk_size=500,chunk_overlap=50",
    )
    assert drifted.is_stale() is True


def test_corpus_fingerprint_changes_with_chunk_text(tmp_path):
    manager = _manager(tmp_path)
    manager.rebuild_feature("Service Model 1", [_record("c1", "Original text.")])
    first = manager.manifest().corpus_sha256

    manager.rebuild_feature("Service Model 1", [_record("c1", "Changed text.")])
    assert manager.manifest().corpus_sha256 != first
