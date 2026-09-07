"""Owns the lifecycle of the global lexical (BM25) index — build, save,
load, rebuild — separately from *querying* it (`BM25Retriever`).

    INDEXING (once, in IndexService.index_feature):
        canonical chunks -> BM25ChunkRecords -> rebuild_feature(feature)
        -> 3 rank_bm25.BM25Okapi built in memory -> corpus.json + manifest.json (atomic)

    STARTUP (FastAPI lifespan) / first query:
        load() -> read corpus.json -> rebuild the 3 BM25Okapi in memory

    QUERY (BM25Retriever.retrieve):
        search(artifact_type, query_tokens, top_k) -> get_scores over the
        pre-built bucket -> term-presence filter -> top-k. Never rebuilds,
        never reads ChromaDB.

Why three buckets keyed by `DocumentCategory` (WORKFLOW / TEST_CASE /
ISSUE) rather than one global index: the only Source-of-Truth BM25
caller (`RetrievalService.retrieve_hybrid`) has always searched one
artifact type at a time with no feature bound, so BM25's IDF has always
been scoped to *one artifact type across every feature*. Precomputing
exactly that — three corpora, each spanning all features — keeps
tokenization, scoring parameters and ranking identical to the old
rebuild-per-query behaviour (same tokens in -> same `BM25Okapi` ->
same `get_scores`). It is not a per-feature physical index: feature
filtering, when a caller asks for it, is a post-scoring filter in
`search`.

The persisted artifact is the tokenized corpus + chunk identity +
metadata + a manifest fingerprint — never a pickled `rank_bm25`
object — so it stays inspectable, portable across deploys, and
checkable against the current corpus/tokenizer/chunking version.

To rebuild the index after Source-of-Truth content, chunking, parsing,
or tokenization changes: re-run indexing for each feature
(`POST /index-feature/{feature}` -> `IndexService.index_feature`), which
calls `rebuild_feature` and re-persists. There is no separate "rebuild
BM25" entry point — it always follows the canonical chunk corpus.
"""
import hashlib
import json
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger
from app.models.bm25_index import BM25BucketStats, BM25ChunkRecord, BM25Manifest
from app.models.common import DocumentCategory
from app.utils.atomic_json import write_json_atomic
from app.utils.datetime_utils import utcnow

logger = get_logger(__name__)

_INDEX_VERSION = 1
_CORPUS_FILENAME = "corpus.json"
_MANIFEST_FILENAME = "manifest.json"

# rank_bm25.BM25Okapi's own defaults — recorded in the manifest so a
# future change to the library's scoring is detectable as staleness.
_BM25_PARAMS: dict[str, str | float] = {"variant": "BM25Okapi", "k1": 1.5, "b": 0.75, "epsilon": 0.25}

_BUCKETS: tuple[DocumentCategory, ...] = (
    DocumentCategory.WORKFLOW,
    DocumentCategory.TEST_CASE,
    DocumentCategory.ISSUE,
)


@dataclass(frozen=True)
class BM25ScoredRecord:
    """One BM25 hit from `BM25IndexManager.search` — the canonical chunk
    record plus its raw BM25 score (kept alongside rank so diagnostics
    can report the actual number)."""

    record: BM25ChunkRecord
    score: float


class _Bucket:
    """One artifact type's in-memory corpus + its built BM25 index."""

    __slots__ = ("records", "bm25")

    def __init__(self, records: list[BM25ChunkRecord]) -> None:
        self.records = records
        # BM25Okapi rejects an empty corpus (division by zero on avgdl);
        # an empty bucket simply has no index and returns nothing.
        self.bm25: BM25Okapi | None = BM25Okapi([r.tokens for r in records]) if records else None


class BM25IndexManager:
    """Thread-safe: `rebuild_feature`/`load`/`save` take a lock and swap
    the in-memory buckets atomically, so a query in flight always sees a
    complete, consistent index (either the old one or the new one).
    """

    def __init__(
        self,
        index_dir: Path,
        collection_name: str,
        tokenizer: Callable[[str], list[str]],
        tokenizer_version: str,
        chunking_signature: str,
    ) -> None:
        self._index_dir = index_dir
        self._collection_name = collection_name
        self._tokenize = tokenizer
        self._tokenizer_version = tokenizer_version
        self._chunking_signature = chunking_signature
        self._lock = threading.RLock()
        self._buckets: dict[DocumentCategory, _Bucket] | None = None
        self._manifest: BM25Manifest | None = None
        self._warned_unavailable = False

    # --- provenance ------------------------------------------------------

    @property
    def index_dir(self) -> Path:
        return self._index_dir

    @property
    def collection_name(self) -> str:
        """The ChromaDB collection these chunks were also embedded into —
        carried onto every `RetrievedChunk` a BM25 hit maps to, exactly
        as the vector path does, so provenance is unchanged."""
        return self._collection_name

    # --- build / persist (indexing time) -------------------------------

    def rebuild_feature(self, feature: str, records: Iterable[BM25ChunkRecord]) -> None:
        """Replaces every row for `feature` (across all three buckets)
        with `records`, rebuilds the affected `BM25Okapi` objects, and
        re-persists — mirroring `VectorStoreService.replace_feature_chunks`
        so BM25 and ChromaDB stay in lock-step through a per-feature
        re-index. An empty `records` simply drops the feature.
        """
        new_records = list(records)
        with self._lock:
            current = self._all_records_by_bucket() if self._buckets is not None else self._read_corpus_or_empty()
            merged: dict[DocumentCategory, list[BM25ChunkRecord]] = {}
            for bucket in _BUCKETS:
                kept = [r for r in current.get(bucket, []) if r.feature != feature]
                added = [r for r in new_records if r.artifact_type == bucket]
                merged[bucket] = kept + added
            self._install(merged, persist=True)

    def rebuild_all(self, records: Iterable[BM25ChunkRecord]) -> None:
        """Replaces the whole index with `records` — every feature at
        once. Not used by the normal per-feature indexing path; handy for
        a one-shot full rebuild or in tests."""
        all_records = list(records)
        with self._lock:
            merged = {bucket: [r for r in all_records if r.artifact_type == bucket] for bucket in _BUCKETS}
            self._install(merged, persist=True)

    def save(self) -> None:
        """Persists the current in-memory index. `rebuild_feature`/
        `rebuild_all` already call this; exposed for completeness."""
        with self._lock:
            if self._buckets is None:
                raise RuntimeError("BM25IndexManager.save called before any index was built or loaded.")
            self._persist({bucket: b.records for bucket, b in self._buckets.items()})

    # --- load / availability (startup / query time) -------------------

    def is_available(self) -> bool:
        """Whether a persisted index exists on disk and parses."""
        return self._read_manifest() is not None and self._corpus_path.is_file()

    def is_loaded(self) -> bool:
        with self._lock:
            return self._buckets is not None

    def load(self) -> None:
        """Reads `corpus.json` from disk and rebuilds the three
        `BM25Okapi` in memory. Raises `FileNotFoundError` if no index has
        been persisted yet — the caller decides whether that's fatal
        (it never is: retrieval degrades to vector-only until indexing
        runs)."""
        with self._lock:
            if not self._corpus_path.is_file():
                raise FileNotFoundError(
                    f"No persistent BM25 index at {self._corpus_path}. Run indexing "
                    "(POST /index-feature/{feature}) to build it."
                )
            raw = json.loads(self._corpus_path.read_text(encoding="utf-8"))
            merged = {
                bucket: [BM25ChunkRecord.model_validate(item) for item in raw.get(bucket.value, [])]
                for bucket in _BUCKETS
            }
            self._buckets = {bucket: _Bucket(records) for bucket, records in merged.items()}
            self._manifest = self._read_manifest()
            self._warned_unavailable = False
            logger.info(
                "Loaded persistent BM25 index from %s: %d chunks (%s).",
                self._index_dir,
                sum(len(b.records) for b in self._buckets.values()),
                ", ".join(f"{bucket.value}={len(self._buckets[bucket].records)}" for bucket in _BUCKETS),
            )

    def ensure_loaded(self) -> bool:
        """Loads the persisted index if it isn't resident yet — a load,
        never a rebuild, never a ChromaDB read. Returns whether an index
        is available to query. Logs the "run indexing" guidance once if
        nothing is persisted."""
        with self._lock:
            if self._buckets is not None:
                return True
            try:
                self.load()
                return True
            except FileNotFoundError:
                if not self._warned_unavailable:
                    logger.warning(
                        "No persistent BM25 index found at %s — lexical retrieval returns nothing until "
                        "indexing is run for each feature (POST /index-feature/{feature}).",
                        self._index_dir,
                    )
                    self._warned_unavailable = True
                return False

    def manifest(self) -> BM25Manifest | None:
        with self._lock:
            return self._manifest or self._read_manifest()

    def is_stale(self) -> bool:
        """Whether the persisted manifest was written with a different
        tokenizer version, chunking signature, or BM25 variant than this
        process expects. Informational only — never forces a rebuild."""
        manifest = self.manifest()
        if manifest is None:
            return False
        return (
            manifest.index_version != _INDEX_VERSION
            or manifest.tokenizer_version != self._tokenizer_version
            or manifest.chunking_signature != self._chunking_signature
            or manifest.bm25_params != _BM25_PARAMS
        )

    # --- query --------------------------------------------------------

    def search(
        self,
        artifact_type: DocumentCategory,
        query_tokens: list[str],
        top_k: int,
        feature: str | None = None,
    ) -> list[BM25ScoredRecord]:
        """Up to `top_k` records from `artifact_type`'s bucket ranked by
        BM25 score against `query_tokens`. A record is included only if
        it shares at least one query term (the raw score can be negative
        for a small corpus — a property of Okapi BM25's IDF, not a bug —
        so inclusion is by term presence, and the possibly-negative score
        still drives ranking and RRF). `feature`, when given, restricts
        results *after* scoring — IDF is unaffected, matching the old
        behaviour where the KA never bound a feature.
        """
        if top_k <= 0 or not query_tokens:
            return []
        with self._lock:
            if self._buckets is None:
                return []
            bucket = self._buckets[artifact_type]
            if bucket.bm25 is None:
                return []
            scores = bucket.bm25.get_scores(query_tokens)
            query_term_set = set(query_tokens)
            scored = [
                BM25ScoredRecord(record=record, score=float(score))
                for record, score in zip(bucket.records, scores, strict=True)
                if query_term_set & set(record.tokens) and (feature is None or record.feature == feature)
            ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    # --- internals --------------------------------------------------

    @property
    def _corpus_path(self) -> Path:
        return self._index_dir / _CORPUS_FILENAME

    @property
    def _manifest_path(self) -> Path:
        return self._index_dir / _MANIFEST_FILENAME

    def _all_records_by_bucket(self) -> dict[DocumentCategory, list[BM25ChunkRecord]]:
        assert self._buckets is not None
        return {bucket: list(b.records) for bucket, b in self._buckets.items()}

    def _read_corpus_or_empty(self) -> dict[DocumentCategory, list[BM25ChunkRecord]]:
        if not self._corpus_path.is_file():
            return {bucket: [] for bucket in _BUCKETS}
        raw = json.loads(self._corpus_path.read_text(encoding="utf-8"))
        return {
            bucket: [BM25ChunkRecord.model_validate(item) for item in raw.get(bucket.value, [])]
            for bucket in _BUCKETS
        }

    def _install(self, merged: dict[DocumentCategory, list[BM25ChunkRecord]], persist: bool) -> None:
        self._buckets = {bucket: _Bucket(records) for bucket, records in merged.items()}
        if persist:
            self._persist(merged)
        self._manifest = self._read_manifest()

    def _persist(self, merged: dict[DocumentCategory, list[BM25ChunkRecord]]) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            bucket.value: [record.model_dump(mode="json") for record in merged[bucket]] for bucket in _BUCKETS
        }
        write_json_atomic(self._corpus_path, payload)
        write_json_atomic(self._manifest_path, self._build_manifest(merged).model_dump(mode="json"))

    def _build_manifest(self, merged: dict[DocumentCategory, list[BM25ChunkRecord]]) -> BM25Manifest:
        all_records = [record for bucket in _BUCKETS for record in merged[bucket]]
        previous = self._read_manifest()
        now = utcnow()
        feature_counts: dict[str, int] = {}
        parser_versions: set[str] = set()
        for record in all_records:
            feature_counts[record.feature] = feature_counts.get(record.feature, 0) + 1
            name = record.metadata.get("parserName")
            version = record.metadata.get("parserVersion")
            if name is not None and version is not None:
                parser_versions.add(f"{name}:{version}")
        return BM25Manifest(
            index_version=_INDEX_VERSION,
            tokenizer_version=self._tokenizer_version,
            chunking_signature=self._chunking_signature,
            bm25_params=_BM25_PARAMS,
            chunk_count=len(all_records),
            buckets={
                bucket.value: BM25BucketStats(
                    chunk_count=len(merged[bucket]), corpus_sha256=_corpus_sha256(merged[bucket])
                )
                for bucket in _BUCKETS
            },
            features=dict(sorted(feature_counts.items())),
            parser_versions=sorted(parser_versions),
            corpus_sha256=_corpus_sha256(all_records),
            created_at=previous.created_at if previous is not None else now,
            updated_at=now,
        )

    def _read_manifest(self) -> BM25Manifest | None:
        if not self._manifest_path.is_file():
            return None
        try:
            return BM25Manifest.model_validate_json(self._manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Persistent BM25 manifest at %s is unreadable; treating index as absent.", self._manifest_path
            )
            return None


def _corpus_sha256(records: Iterable[BM25ChunkRecord]) -> str:
    """A deterministic fingerprint of a chunk set — order-independent
    (records sorted by id), text-sensitive — so a persisted index can be
    checked against a freshly-chunked corpus."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.chunk_id):
        digest.update(record.chunk_id.encode("utf-8"))
        digest.update(b"\t")
        digest.update(record.text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
