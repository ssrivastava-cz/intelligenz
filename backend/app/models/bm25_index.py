"""Persisted shapes for the global lexical (BM25) index.

The persistent artifact is deliberately *not* a pickled `rank_bm25`
object — it is the tokenized corpus plus each chunk's identity and
metadata (`BM25ChunkRecord`), plus a `BM25Manifest` fingerprint. A
`rank_bm25.BM25Okapi` is rebuilt deterministically from the stored
tokens on load, so the on-disk format stays inspectable, portable
across deploys, and safe to validate against the current corpus /
tokenizer / chunking version.

See `app.services.bm25_index_manager.BM25IndexManager`.
"""
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.common import DocumentCategory

MetadataValue = str | int | float | bool


class BM25ChunkRecord(BaseModel):
    """One canonical chunk as the BM25 index needs it — the same unit
    that was embedded into ChromaDB, carrying enough to map a BM25 hit
    straight back to a `RetrievedChunk` without a second chunk
    representation.

    `metadata` is the exact dict `build_chunk_metadata` produced for
    this chunk (what ChromaDB stores), so `chunk_mapping`-style
    attribute extraction works identically on a BM25 hit and a vector
    hit. `tokens` is `app.retrievers.text_tokenizer.tokenize(text)`
    captured at index-build time, so loading the index never re-runs
    tokenization and the persisted corpus is self-contained.
    """

    chunk_id: str
    text: str
    feature: str
    artifact_type: DocumentCategory
    metadata: dict[str, MetadataValue]
    tokens: list[str]

    @classmethod
    def from_chunk(
        cls,
        chunk_id: str,
        text: str,
        metadata: dict[str, MetadataValue],
        tokens: list[str],
    ) -> "BM25ChunkRecord":
        return cls(
            chunk_id=chunk_id,
            text=text,
            feature=str(metadata["feature"]),
            artifact_type=DocumentCategory(str(metadata["artifactType"])),
            metadata=dict(metadata),
            tokens=tokens,
        )


class BM25BucketStats(BaseModel):
    chunk_count: int
    corpus_sha256: str


class BM25Manifest(BaseModel):
    """Lightweight fingerprint of a persisted BM25 index — enough to
    detect a stale or incompatible index (different corpus, tokenizer,
    chunking parameters, or BM25 variant) without loading the corpus.
    Never triggers an automatic rebuild; it only informs logs and
    `BM25IndexManager.is_stale`.
    """

    index_version: int
    tokenizer_version: str
    chunking_signature: str
    bm25_params: dict[str, MetadataValue]
    chunk_count: int
    buckets: dict[str, BM25BucketStats] = Field(default_factory=dict)
    features: dict[str, int] = Field(default_factory=dict)
    parser_versions: list[str] = Field(default_factory=list)
    corpus_sha256: str
    created_at: datetime
    updated_at: datetime
