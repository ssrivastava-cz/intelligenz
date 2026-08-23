"""Reranking — scores each hybrid candidate against the user's *exact*
question, independent of whichever retrieval method(s) found it. This is
the stage that should catch what pure vector similarity misses: a
generic, heavily-cross-linked section ("Related articles") can sit close
to almost everything in embedding space without actually answering any
specific question.

`Reranker` is an interface specifically so the scoring model is
swappable without touching anything else in the pipeline (see
`app.core.dependencies.get_reranker`). The default implementation,
`LexicalReranker`, is a local, dependency-light (pure Python + numpy,
already installed) IDF-weighted term-overlap scorer — deliberately not a
neural cross-encoder. A real cross-encoder (e.g. `sentence-transformers`
with `cross-encoder/ms-marco-MiniLM-L-6-v2`) would score relevance more
precisely by jointly attending over the (question, chunk) pair, but pulls
in ~600MB-1GB of new dependencies (torch), a ~90MB model download cached
on first use, and materially higher per-request CPU latency — a real
infrastructure decision, not something to add silently. `LexicalReranker`
needs none of that: no model download, sub-millisecond per candidate,
fully deterministic (good for tests), and well-suited to this domain,
which is full of exact terminology (ticket/test-case IDs, UI labels,
feature names) that lexical overlap catches directly.
"""
import math
from abc import ABC, abstractmethod

from app.retrievers.rank_fusion import FusedCandidate
from app.retrievers.text_tokenizer import tokenize

# Generic, heavily cross-linked sections that tend to sit close to
# almost everything in embedding space without answering any specific
# question — exactly the failure mode reported ("Related articles"
# outranking the chunk that actually answers the question). A candidate
# under one of these headings is never excluded outright (real content
# can still live there), just no longer given the benefit of the doubt.
_BOILERPLATE_SECTION_HEADINGS = frozenset(
    {"related articles", "related links", "see also", "references", "further reading"}
)
_BOILERPLATE_PENALTY_MULTIPLIER = 0.5

# How much a single exact multi-word phrase match (e.g. the question
# literally contains "Contact Log" and so does the chunk) can add to the
# term-overlap score — capped so it can influence, but never alone
# decide, a candidate's rank.
_PHRASE_MATCH_BONUS = 0.2


class Reranker(ABC):
    @abstractmethod
    def score(self, query_text: str, candidates: list[FusedCandidate]) -> None:
        """Sets `candidate.diagnostic.reranker_score` (0-1, higher is
        more relevant) on every candidate in place — mutating the same
        `HybridCandidate` diagnostic objects `ReciprocalRankFusion`
        already built, rather than returning a second, parallel
        structure that could drift out of alignment with `candidates`.
        """


class LexicalReranker(Reranker):
    def score(self, query_text: str, candidates: list[FusedCandidate]) -> None:
        query_terms = tokenize(query_text)
        if not query_terms:
            for candidate in candidates:
                candidate.diagnostic.reranker_score = 0.0
            return

        candidate_terms = [tokenize(candidate.chunk.text) for candidate in candidates]
        idf, default_idf = _inverse_document_frequency(candidate_terms)
        query_phrases = _adjacent_word_pairs(query_text)

        for candidate, terms in zip(candidates, candidate_terms, strict=True):
            candidate.diagnostic.reranker_score = _score_one(
                query_terms,
                set(terms),
                query_phrases,
                candidate.chunk.text,
                candidate.chunk.section_heading,
                idf,
                default_idf,
            )


def _score_one(
    query_terms: list[str],
    chunk_term_set: set[str],
    query_phrases: list[str],
    chunk_text: str,
    section_heading: str | None,
    idf: dict[str, float],
    default_idf: float,
) -> float:
    # A query term that never appears in *any* candidate gets the
    # maximum possible IDF (as if it were the rarest possible term) —
    # the same formula `_inverse_document_frequency` uses for a term
    # with zero document frequency, just not materialized in the dict
    # since it was never observed in the candidate pool at all.
    total_weight = sum(idf.get(term, default_idf) for term in query_terms)
    matched_weight = sum(idf[term] for term in query_terms if term in chunk_term_set)
    overlap_score = (matched_weight / total_weight) if total_weight > 0 else 0.0

    chunk_text_lower = chunk_text.lower()
    phrase_bonus = _PHRASE_MATCH_BONUS if any(phrase in chunk_text_lower for phrase in query_phrases) else 0.0

    score = min(overlap_score + phrase_bonus, 1.0)

    if section_heading and section_heading.strip().lower() in _BOILERPLATE_SECTION_HEADINGS:
        score *= _BOILERPLATE_PENALTY_MULTIPLIER

    return score


def _inverse_document_frequency(documents: list[list[str]]) -> tuple[dict[str, float], float]:
    """Smoothed IDF computed over just this request's own candidate
    pool (not a corpus-wide index) — a term common across *this
    question's* candidates (e.g. the feature name, present in nearly
    every chunk retrieved for it) is weighted down relative to a term
    that discriminates between them, exactly like standard IDF, just
    scoped to what's actually being ranked. Also returns the maximum
    possible IDF (the score of a term with zero observed document
    frequency) for terms that never appear in the pool at all.
    """
    total_documents = len(documents)
    document_frequency: dict[str, int] = {}
    for terms in documents:
        for term in set(terms):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    idf = {
        term: math.log((total_documents + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }
    default_idf = math.log(total_documents + 1) + 1.0
    return idf, default_idf


def _adjacent_word_pairs(text: str) -> list[str]:
    """Lowercased two-word phrases from the *original* text (before
    stopword removal or stemming) — catches an exact multi-word entity
    like "Contact Log" or "Encounter Status" as a single unit, which
    word-level overlap alone would only give partial credit for.
    """
    words = [word.lower() for word in text.split() if word]
    return [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
