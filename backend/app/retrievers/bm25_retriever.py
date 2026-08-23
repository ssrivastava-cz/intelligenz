"""Lexical (keyword) retrieval — the second leg of Hybrid Retrieval
alongside vector search. Runs entirely locally via `rank_bm25`: no
OpenAI call, no network request, no model download. Exists specifically
for exact terminology a bi-encoder embedding can under-rank ("C_10",
"Contact Log", issue/test-case IDs, specific UI labels) — see
`app.retrievers.rank_fusion` for how its ranking is combined with the
vector search's.

Builds its BM25 index fresh on every call, over whatever
`VectorStoreService.get_chunks_for_filter` returns for the requested
scope (typically one feature + one artifact type) — there is no
persisted BM25 index to keep in sync with ChromaDB, so this can never
drift from what's actually indexed. For the Source of Truth corpus
sizes this app indexes (tens to low hundreds of chunks per feature),
rebuilding per-request is cheap; see `RetrievalDiagnostics.bm25RetrievalMs`
for the actual measured cost.
"""
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.chroma_filters import build_where_clause
from app.retrievers.text_tokenizer import tokenize
from app.services.vector_store_service import VectorMatch, VectorStoreService


@dataclass(frozen=True)
class BM25Match:
    """One BM25 hit — the chunk plus its raw BM25 score, kept alongside
    (not just implied by rank) so `RetrievalDiagnostics` can report the
    actual score, not just a position.
    """

    chunk: RetrievedChunk
    score: float


class BM25Retriever:
    def __init__(self, vector_store: VectorStoreService, feature: str) -> None:
        self._vector_store = vector_store
        self._feature = feature

    def retrieve(self, query_text: str, top_k: int, filters: dict[str, str] | None = None) -> list[BM25Match]:
        """Returns up to `top_k` chunks ranked by BM25 score against
        `query_text`, restricted to this instance's `feature` and any
        additional `filters` (e.g. artifact type) — mirroring
        `Retriever.retrieve`'s filter contract, just keyed by query text
        instead of a query embedding.
        """
        if top_k <= 0:
            return []

        where = {"feature": self._feature, **(filters or {})}
        corpus = self._vector_store.get_chunks_for_filter(where=build_where_clause(where))
        if not corpus:
            return []

        query_terms = tokenize(query_text)
        if not query_terms:
            return []
        query_term_set = set(query_terms)

        tokenized_corpus = [tokenize(match.chunk_text) for match in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query_terms)

        # A candidate is only included if it actually contains at least
        # one query term — *not* filtered by the raw BM25 score being
        # positive. Okapi BM25's IDF term goes negative for a word that
        # appears in most/all documents of a small corpus (a well-known
        # property of the classic formula, not a bug), which would
        # otherwise wrongly exclude a real, literal term match just
        # because the corpus happened to be small. The (possibly
        # negative) score is still reported and still determines
        # ranking/RRF contribution — only inclusion is decided by actual
        # term presence.
        candidates = [
            (match, score, terms)
            for match, score, terms in zip(corpus, scores, tokenized_corpus, strict=True)
            if query_term_set & set(terms)
        ]
        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
        return [
            BM25Match(chunk=_to_retrieved_chunk(match, self._vector_store.collection_name), score=float(score))
            for match, score, _terms in ranked[:top_k]
        ]


def _to_retrieved_chunk(match: VectorMatch, collection_name: str) -> RetrievedChunk:
    """Distinct from `app.retrievers.chunk_mapping.to_retrieved_chunk`
    on purpose: that one derives `similarity_score` from a real vector
    distance, which this match never has (`VectorMatch.distance` is
    always `0.0` for a metadata-only read) — mapping it through the same
    `distance_to_similarity` formula would fabricate a perfect `1.0`
    vector similarity for a chunk that was never vector-compared at all.
    `similarity_score` is `0.0` here; `HybridCandidate.bm25_score` (set
    by the caller) is where this chunk's actual relevance signal lives.
    """
    metadata = match.metadata
    return RetrievedChunk(
        chunk_id=match.chunk_id,
        text=match.chunk_text,
        similarity_score=0.0,
        vector_distance=0.0,
        artifact_type=metadata["artifactType"],
        feature=metadata["feature"],
        source_filename=metadata["sourceFilename"],
        section_heading=metadata.get("sectionHeading"),
        page_number=metadata.get("pageNumber"),
        collection_name=collection_name,
        document_id=metadata.get("documentId"),
        chunk_number=metadata.get("chunkNumber"),
    )
