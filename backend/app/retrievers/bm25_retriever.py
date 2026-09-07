"""Lexical (keyword) retrieval — the second leg of Hybrid Retrieval
alongside vector search. Exists specifically for exact terminology a
bi-encoder embedding can under-rank ("C_10", "Contact Log", issue/
test-case IDs, specific UI labels) — see `app.retrievers.rank_fusion`
for how its ranking is combined with the vector search's.

Query-only. The BM25 index is built once during the indexing pipeline
(`IndexService.index_feature` -> `BM25IndexManager.rebuild_feature`) and
persisted; this retriever loads it (a load, never a rebuild) and runs
`get_scores` against it:

    query -> BM25IndexManager.search(artifact_type, tokens, top_k) -> BM25Match -> RRF

`retrieve()` never traverses Source of Truth, never parses or chunks,
never reads the whole ChromaDB collection, and never reconstructs the
index. If no index has been persisted yet it returns `[]` (logged once,
with the "run indexing" guidance) so hybrid retrieval degrades to
vector-only rather than failing.

Tokenization and the term-presence inclusion rule are unchanged from the
previous rebuild-per-query implementation; see `BM25IndexManager.search`.
Uploaded documents use `app.retrievers.ephemeral_bm25` instead — their
per-session collections are not part of this persistent index.
"""
from dataclasses import dataclass
from typing import Any

from app.models.common import DocumentCategory
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.text_tokenizer import tokenize
from app.services.bm25_index_manager import BM25IndexManager, BM25ScoredRecord


@dataclass(frozen=True)
class BM25Match:
    """One BM25 hit — the chunk plus its raw BM25 score, kept alongside
    (not just implied by rank) so `RetrievalDiagnostics` can report the
    actual score, not just a position.
    """

    chunk: RetrievedChunk
    score: float


class BM25Retriever:
    def __init__(self, index_manager: BM25IndexManager, feature: str | None = None) -> None:
        self._index_manager = index_manager
        self._feature = feature

    def retrieve(self, query_text: str, top_k: int, filters: dict[str, str] | None = None) -> list[BM25Match]:
        """Returns up to `top_k` chunks ranked by BM25 score against
        `query_text`, restricted to this instance's `feature` (when one
        was bound at construction — `feature=None` searches every
        feature's chunks, which is how the Knowledge Assistant uses it)
        and the `artifactType` in `filters` — mirroring the old
        signature exactly, just served from the persistent index.
        """
        if top_k <= 0:
            return []
        if not self._index_manager.ensure_loaded():
            return []

        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        artifact_types = _artifact_types(filters)
        scored: list[BM25ScoredRecord] = []
        for artifact_type in artifact_types:
            scored.extend(
                self._index_manager.search(artifact_type, query_tokens, top_k, feature=self._feature)
            )
        scored.sort(key=lambda item: item.score, reverse=True)

        collection_name = self._index_manager.collection_name
        return [
            BM25Match(chunk=_to_retrieved_chunk(item, collection_name), score=item.score)
            for item in scored[:top_k]
        ]


def _artifact_types(filters: dict[str, str] | None) -> tuple[DocumentCategory, ...]:
    """The one bucket named by an `artifactType` filter, or all three
    when none is given. Every current Source-of-Truth caller passes one
    (`RetrievalService.retrieve_hybrid` queries per artifact type)."""
    if filters and "artifactType" in filters:
        return (DocumentCategory(filters["artifactType"]),)
    return (DocumentCategory.WORKFLOW, DocumentCategory.TEST_CASE, DocumentCategory.ISSUE)


def _to_retrieved_chunk(item: BM25ScoredRecord, collection_name: str) -> RetrievedChunk:
    """Maps a persisted `BM25ChunkRecord` to `RetrievedChunk` exactly as
    the old `VectorMatch`-based mapping did — `similarity_score`/
    `vector_distance` are `0.0` (this chunk was never vector-compared;
    fabricating a `1.0` similarity would misreport it), and the chunk's
    real lexical signal lives in `HybridCandidate.bm25_score`, set by the
    caller. `metadata` is the same dict shape ChromaDB stores, so the
    attribute extraction is identical to `chunk_mapping.to_retrieved_chunk`.
    """
    # `dict[str, Any]` (not the stored `MetadataValue` union) so the
    # per-key extraction below type-checks — each Chroma metadata field's
    # real type is known here even though the dict is heterogeneous.
    metadata: dict[str, Any] = item.record.metadata
    return RetrievedChunk(
        chunk_id=item.record.chunk_id,
        text=item.record.text,
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
