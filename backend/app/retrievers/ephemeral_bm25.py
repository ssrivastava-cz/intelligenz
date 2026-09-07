"""Rebuild-per-query lexical search over a ChromaDB collection — the
original `BM25Retriever` algorithm, now used *only* for uploaded
documents.

Uploaded documents live in short-lived, per-session ChromaDB
collections that are never part of the persistent global Source-of-Truth
BM25 index (`app.services.bm25_index_manager`). For those, rebuilding a
tiny `BM25Okapi` from the session's chunks on each query is still the
simplest correct approach and cannot drift from what's indexed. The
Source-of-Truth path, by contrast, queries the persistent index via
`app.retrievers.bm25_retriever.BM25Retriever`.
"""
from typing import Any

from rank_bm25 import BM25Okapi

from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.bm25_retriever import BM25Match
from app.retrievers.chroma_filters import build_where_clause
from app.retrievers.text_tokenizer import tokenize
from app.services.vector_store_service import VectorMatch, VectorStoreService


def ephemeral_bm25_search(
    vector_store: VectorStoreService,
    query_text: str,
    top_k: int,
    feature: str | None = None,
    filters: dict[str, str] | None = None,
) -> list[BM25Match]:
    """Up to `top_k` chunks from `vector_store` ranked by BM25 against
    `query_text`, restricted to `feature` (when given) plus any extra
    `filters`. Builds the BM25 index fresh from whatever the filter
    matches — appropriate only for the small, session-scoped upload
    collections this is used for.
    """
    if top_k <= 0:
        return []

    feature_filter = {"feature": feature} if feature is not None else {}
    where = {**feature_filter, **(filters or {})}
    corpus = vector_store.get_chunks_for_filter(where=build_where_clause(where))
    if not corpus:
        return []

    query_terms = tokenize(query_text)
    if not query_terms:
        return []
    query_term_set = set(query_terms)

    tokenized_corpus = [tokenize(match.chunk_text) for match in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_terms)

    # Inclusion is by literal term presence, not by the raw score being
    # positive: Okapi BM25's IDF term goes negative for a word present in
    # most/all documents of a small corpus (a known property of the
    # classic formula), which would otherwise wrongly drop a real literal
    # match. The (possibly negative) score still drives ranking/RRF.
    candidates = [
        (match, score, terms)
        for match, score, terms in zip(corpus, scores, tokenized_corpus, strict=True)
        if query_term_set & set(terms)
    ]
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    return [
        BM25Match(chunk=_to_retrieved_chunk(match, vector_store.collection_name), score=float(score))
        for match, score, _terms in ranked[:top_k]
    ]


def _to_retrieved_chunk(match: VectorMatch, collection_name: str) -> RetrievedChunk:
    # `dict[str, Any]` (not the stored `MetadataValue` union) so the
    # per-key narrowing below type-checks — a Chroma metadata dict is
    # heterogeneous by nature and each field's real type is known here.
    metadata: dict[str, Any] = match.metadata
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
        document_title=metadata.get("documentTitle"),
        source_path=metadata.get("sourcePath"),
        source_folder=metadata.get("sourceFolder"),
    )
