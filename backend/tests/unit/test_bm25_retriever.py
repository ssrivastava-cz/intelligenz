"""Unit tests for `BM25Retriever` — now a *query-only* wrapper over the
persistent `BM25IndexManager`. It must never rebuild the index and never
read a ChromaDB collection; those responsibilities moved to indexing
time (`IndexService.index_feature` -> `BM25IndexManager.rebuild_feature`).
"""
from app.models.bm25_index import BM25ChunkRecord
from app.models.common import DocumentCategory
from app.retrievers.bm25_retriever import BM25Retriever
from app.retrievers.text_tokenizer import TOKENIZER_VERSION, tokenize
from app.services.bm25_index_manager import BM25IndexManager

_COLLECTION = "source_of_truth_chunks"


def _manager(tmp_path) -> BM25IndexManager:
    return BM25IndexManager(
        index_dir=tmp_path / "bm25",
        collection_name=_COLLECTION,
        tokenizer=tokenize,
        tokenizer_version=TOKENIZER_VERSION,
        chunking_signature="chunk_size=500,chunk_overlap=50",
    )


def _record(chunk_id, text, *, feature="Contact Log", artifact_type="WORKFLOW", **metadata) -> BM25ChunkRecord:
    meta = {
        "chunkId": chunk_id,
        "artifactType": artifact_type,
        "feature": feature,
        "documentSource": "source_of_truth",
        "sourceFilename": "Contact_Log_Workflow.md",
        "parserName": "MarkdownParser",
        "parserVersion": "1.0",
        **metadata,
    }
    return BM25ChunkRecord.from_chunk(chunk_id, text, meta, tokenize(text))


def _loaded_manager(tmp_path, records) -> BM25IndexManager:
    _manager(tmp_path).rebuild_all(records)
    manager = _manager(tmp_path)
    manager.load()
    return manager


def test_retrieve_returns_empty_when_no_index_has_been_built(tmp_path):
    retriever = BM25Retriever(_manager(tmp_path / "empty"))

    assert retriever.retrieve("Contact Log", top_k=10) == []


def test_retrieve_finds_a_chunk_containing_exact_terminology(tmp_path):
    manager = _loaded_manager(
        tmp_path,
        [
            _record("chunk-1", "The C_10 field controls encounter status transitions for a contact."),
            _record("chunk-2", "General onboarding steps for new users of the platform."),
        ],
    )
    retriever = BM25Retriever(manager)

    matches = retriever.retrieve("What does C_10 control?", top_k=10, filters={"artifactType": "WORKFLOW"})

    assert [m.chunk.chunk_id for m in matches] == ["chunk-1"]
    assert matches[0].chunk.collection_name == _COLLECTION


def test_retrieve_ranks_the_more_lexically_relevant_chunk_first(tmp_path):
    manager = _loaded_manager(
        tmp_path,
        [
            _record("exact", "Delete and Undo Delete both apply to a Contact Log entry directly."),
            _record("tangential", "Contact Log is one of many features available in the product."),
        ],
    )

    matches = BM25Retriever(manager).retrieve(
        "How do I delete and undo delete a Contact Log entry?", top_k=10, filters={"artifactType": "WORKFLOW"}
    )

    assert matches[0].chunk.chunk_id == "exact"


def test_retrieve_respects_the_artifact_type_filter(tmp_path):
    manager = _loaded_manager(
        tmp_path,
        [
            _record("workflow-chunk", "Encounter Status workflow details.", artifact_type="WORKFLOW"),
            _record("test-case-chunk", "Encounter Status test case steps.", artifact_type="TEST_CASE"),
        ],
    )

    matches = BM25Retriever(manager).retrieve("Encounter Status", top_k=10, filters={"artifactType": "WORKFLOW"})

    assert [m.chunk.chunk_id for m in matches] == ["workflow-chunk"]


def test_retrieve_with_a_bound_feature_excludes_other_features(tmp_path):
    manager = _loaded_manager(
        tmp_path,
        [
            _record("here", "Contact Log bridging behaviour.", feature="Contact Log"),
            _record("elsewhere", "Contact Log bridging behaviour.", feature="Other Feature"),
        ],
    )

    bound = BM25Retriever(manager, feature="Contact Log")
    unbound = BM25Retriever(manager)

    assert [m.chunk.chunk_id for m in bound.retrieve("Contact Log bridging", 10, {"artifactType": "WORKFLOW"})] == [
        "here"
    ]
    assert {m.chunk.chunk_id for m in unbound.retrieve("Contact Log bridging", 10, {"artifactType": "WORKFLOW"})} == {
        "here",
        "elsewhere",
    }


def test_retrieve_returns_no_more_than_top_k(tmp_path):
    manager = _loaded_manager(
        tmp_path,
        [_record(cid, f"Contact Log detail {cid}.") for cid in ("a", "b", "c")],
    )

    matches = BM25Retriever(manager).retrieve("Contact Log detail", top_k=2, filters={"artifactType": "WORKFLOW"})

    assert len(matches) <= 2


def test_retrieve_returns_empty_for_a_stopword_only_query(tmp_path):
    manager = _loaded_manager(tmp_path, [_record("chunk-1", "Contact Log workflow details.")])

    assert BM25Retriever(manager).retrieve("how does the", top_k=10, filters={"artifactType": "WORKFLOW"}) == []


def test_retrieve_exposes_full_source_attribution(tmp_path):
    manager = _loaded_manager(
        tmp_path,
        [
            _record(
                "chunk-1",
                "The C_10 field controls encounter status transitions for a contact.",
                sectionHeading="Encounter Status",
                pageNumber=2,
                documentTitle="Contact Log Workflow Guide",
                sourcePath="source_of_truth/Contact and Sticket Log/workflows/Contact_Log_Workflow.md",
                sourceFolder="Contact and Sticket Log",
                documentId="doc-9",
                chunkNumber=4,
            )
        ],
    )

    [match] = BM25Retriever(manager).retrieve("C_10", top_k=10, filters={"artifactType": "WORKFLOW"})

    chunk = match.chunk
    assert chunk.section_heading == "Encounter Status"
    assert chunk.page_number == 2
    assert chunk.document_title == "Contact Log Workflow Guide"
    assert chunk.source_path == "source_of_truth/Contact and Sticket Log/workflows/Contact_Log_Workflow.md"
    assert chunk.source_folder == "Contact and Sticket Log"
    assert chunk.document_id == "doc-9"
    assert chunk.chunk_number == 4
    # never a fabricated vector similarity for a chunk that was never vector-compared
    assert chunk.similarity_score == 0.0
    assert chunk.vector_distance == 0.0


def test_retrieve_does_not_rebuild_the_index(tmp_path, monkeypatch):
    """E: retrieve() queries the loaded index and never triggers a build."""
    manager = _loaded_manager(tmp_path, [_record("chunk-1", "Contact Log bridging behaviour.")])

    def _boom(*_args, **_kwargs):
        raise AssertionError("retrieve() must not rebuild the BM25 index")

    monkeypatch.setattr(manager, "rebuild_feature", _boom)
    monkeypatch.setattr(manager, "rebuild_all", _boom)

    matches = BM25Retriever(manager).retrieve("Contact Log bridging", 10, {"artifactType": "WORKFLOW"})

    assert [m.chunk.chunk_id for m in matches] == ["chunk-1"]


def test_retriever_holds_no_vector_store(tmp_path):
    """F: structurally impossible for retrieve() to read the ChromaDB
    corpus to build BM25 — it has no vector store reference at all."""
    retriever = BM25Retriever(_loaded_manager(tmp_path, [_record("chunk-1", "text")]))

    assert not hasattr(retriever, "_vector_store")


def test_retrieve_without_artifact_type_filter_searches_all_buckets(tmp_path):
    manager = _loaded_manager(
        tmp_path,
        [
            _record("w1", "Eligibility workflow.", artifact_type="WORKFLOW"),
            _record("t1", "Eligibility test case.", artifact_type="TEST_CASE"),
            _record("i1", "Eligibility issue.", artifact_type="ISSUE"),
        ],
    )

    matches = BM25Retriever(manager).retrieve("eligibility", top_k=10)

    assert {m.chunk.chunk_id for m in matches} == {"w1", "t1", "i1"}


def test_retrieve_maps_artifact_type_enum_onto_the_chunk(tmp_path):
    manager = _loaded_manager(tmp_path, [_record("t1", "Eligibility test case steps.", artifact_type="TEST_CASE")])

    [match] = BM25Retriever(manager).retrieve("eligibility test", 10, {"artifactType": "TEST_CASE"})

    assert match.chunk.artifact_type == DocumentCategory.TEST_CASE
