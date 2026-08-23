"""Unit tests for `LexicalReranker` — scores hybrid candidates against
the user's exact question, independent of whichever retrieval method
found them. See `app.retrievers.reranker`.
"""
from app.models.hybrid_retrieval import HybridCandidate
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.rank_fusion import FusedCandidate
from app.retrievers.reranker import LexicalReranker


def _candidate(chunk_id: str, text: str, section_heading: str | None = None) -> FusedCandidate:
    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        similarity_score=0.1,
        artifact_type="WORKFLOW",
        feature="Contact Log",
        source_filename="a.md",
        section_heading=section_heading,
        page_number=None,
        collection_name="source_of_truth_chunks",
    )
    return FusedCandidate(
        chunk=chunk,
        diagnostic=HybridCandidate(
            chunk_id=chunk_id,
            document_id=None,
            document_name="a.md",
            artifact_type="WORKFLOW",
            section_heading=section_heading,
        ),
    )


def test_score_sets_reranker_score_on_every_candidate():
    candidates = [_candidate("a", "Contact Log details."), _candidate("b", "Something else entirely.")]
    reranker = LexicalReranker()

    reranker.score("How does Contact Log work?", candidates)

    assert all(candidate.diagnostic.reranker_score is not None for candidate in candidates)


def test_a_highly_relevant_chunk_outranks_a_topically_related_but_irrelevant_chunk():
    """The exact scenario reported: a generic section can share a topic
    (embedding-space proximity) without answering the specific question
    — the reranker must still separate them via term-level relevance."""
    relevant = _candidate(
        "relevant",
        "Delete and Undo Delete both apply directly to a Contact Log entry, restoring it within 24 hours.",
    )
    boilerplate = _candidate(
        "boilerplate",
        "For more information, browse our knowledge base and other related articles about the product.",
        section_heading="Related Articles",
    )
    reranker = LexicalReranker()
    candidates = [boilerplate, relevant]

    reranker.score("How do I delete and undo delete a Contact Log entry?", candidates)

    assert relevant.diagnostic.reranker_score > boilerplate.diagnostic.reranker_score


def test_a_boilerplate_section_heading_is_penalized():
    same_text_relevant_heading = _candidate("a", "Contact Log workflow details.", section_heading="Contact Log")
    same_text_boilerplate_heading = _candidate("b", "Contact Log workflow details.", section_heading="Related Articles")
    reranker = LexicalReranker()
    candidates = [same_text_relevant_heading, same_text_boilerplate_heading]

    reranker.score("Contact Log workflow", candidates)

    relevant_score = same_text_relevant_heading.diagnostic.reranker_score
    boilerplate_score = same_text_boilerplate_heading.diagnostic.reranker_score
    assert relevant_score > boilerplate_score


def test_an_exact_phrase_match_scores_higher_than_scattered_individual_words():
    exact_phrase = _candidate("exact", "The Contact Log feature lets users bridge duplicate entries.")
    scattered_words = _candidate(
        "scattered", "Users can log an issue in the system. Someone should contact support about the outcome."
    )
    reranker = LexicalReranker()
    candidates = [scattered_words, exact_phrase]

    reranker.score("How does Contact Log work?", candidates)

    assert exact_phrase.diagnostic.reranker_score > scattered_words.diagnostic.reranker_score


def test_score_is_zero_for_a_chunk_with_no_term_overlap_at_all():
    unrelated = _candidate("unrelated", "Pricing tiers for the enterprise subscription plan.")
    reranker = LexicalReranker()
    candidates = [unrelated]

    reranker.score("How does Contact Log work?", candidates)

    assert unrelated.diagnostic.reranker_score == 0.0


def test_score_handles_an_empty_candidate_list():
    reranker = LexicalReranker()

    reranker.score("How does Contact Log work?", [])  # must not raise


def test_scores_are_bounded_between_zero_and_one():
    candidates = [_candidate("a", "Contact Log Contact Log Contact Log workflow details for Contact Log.")]
    reranker = LexicalReranker()

    reranker.score("Contact Log", candidates)

    assert 0.0 <= candidates[0].diagnostic.reranker_score <= 1.0
