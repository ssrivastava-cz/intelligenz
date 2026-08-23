"""Unit tests for `ChunkDeduplicator` — removes near-duplicate chunks
(e.g. the same information repeated across `Contact_Workflow.txt` and
`Sticket_Workflow.txt`) before they reach the prompt. See
`app.retrievers.chunk_deduplicator`.
"""
from app.models.hybrid_retrieval import HybridCandidate
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.chunk_deduplicator import ChunkDeduplicator
from app.retrievers.rank_fusion import FusedCandidate


def _candidate(chunk_id: str, text: str, reranker_score: float, document_name: str = "a.md") -> FusedCandidate:
    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        similarity_score=0.1,
        artifact_type="WORKFLOW",
        feature="Contact Log",
        source_filename=document_name,
        section_heading=None,
        page_number=None,
        collection_name="source_of_truth_chunks",
    )
    return FusedCandidate(
        chunk=chunk,
        diagnostic=HybridCandidate(
            chunk_id=chunk_id,
            document_id=None,
            document_name=document_name,
            artifact_type="WORKFLOW",
            section_heading=None,
            reranker_score=reranker_score,
        ),
    )


def test_deduplicate_returns_empty_list_for_no_candidates():
    deduplicator = ChunkDeduplicator()

    kept, removed = deduplicator.deduplicate([])

    assert kept == []
    assert removed == 0


def test_deduplicate_keeps_two_genuinely_different_chunks():
    a = _candidate("a", "Contact Log lets you bridge duplicate contacts.", reranker_score=0.9)
    b = _candidate("b", "Encounter Status controls appointment lifecycle transitions.", reranker_score=0.8)
    deduplicator = ChunkDeduplicator()

    kept, removed = deduplicator.deduplicate([a, b])

    assert [c.chunk.chunk_id for c in kept] == ["a", "b"]
    assert removed == 0


def test_deduplicate_removes_a_near_duplicate_chunk_and_keeps_the_higher_ranked_one():
    """Mirrors the Contact_Workflow / Sticket_Workflow overlap scenario
    — the same information, worded almost identically, appearing in two
    different source documents."""
    higher_ranked = _candidate(
        "contact-workflow",
        "Bridged contacts are merged automatically when matching criteria align across records.",
        reranker_score=0.9,
        document_name="Contact_Workflow.txt",
    )
    near_duplicate = _candidate(
        "sticket-workflow",
        "Bridged contacts are merged automatically when matching criteria align across records.",
        reranker_score=0.6,
        document_name="Sticket_Workflow.txt",
    )
    deduplicator = ChunkDeduplicator()

    # Candidates must already be sorted best-first — the first
    # (highest-ranked) representative of a duplicate group is kept.
    kept, removed = deduplicator.deduplicate([higher_ranked, near_duplicate])

    assert [c.chunk.chunk_id for c in kept] == ["contact-workflow"]
    assert removed == 1


def test_deduplicate_removes_exact_duplicates():
    a = _candidate("a", "Identical text here.", reranker_score=0.9)
    b = _candidate("b", "Identical text here.", reranker_score=0.5)
    deduplicator = ChunkDeduplicator()

    kept, removed = deduplicator.deduplicate([a, b])

    assert [c.chunk.chunk_id for c in kept] == ["a"]
    assert removed == 1


def test_deduplicate_preserves_legitimate_adjacent_context_chunks():
    """Two chunks from the same document, covering different content —
    not near-duplicates just because they're neighbours."""
    step_one = _candidate("step-1", "Step 1: Open the Contact Log entry and select Edit.", reranker_score=0.9)
    step_two = _candidate("step-2", "Step 2: Update the encounter status field and save.", reranker_score=0.8)
    deduplicator = ChunkDeduplicator()

    kept, removed = deduplicator.deduplicate([step_one, step_two])

    assert len(kept) == 2
    assert removed == 0


def test_deduplicate_respects_a_custom_similarity_threshold():
    a = _candidate("a", "Contact Log workflow overview and general details.", reranker_score=0.9)
    b = _candidate("b", "Contact Log workflow summary and general information.", reranker_score=0.7)
    strict_deduplicator = ChunkDeduplicator(similarity_threshold=0.99)
    lenient_deduplicator = ChunkDeduplicator(similarity_threshold=0.2)

    strict_kept, strict_removed = strict_deduplicator.deduplicate([a, b])
    lenient_kept, lenient_removed = lenient_deduplicator.deduplicate([a, b])

    assert strict_removed == 0
    assert len(strict_kept) == 2
    assert lenient_removed == 1
    assert len(lenient_kept) == 1
