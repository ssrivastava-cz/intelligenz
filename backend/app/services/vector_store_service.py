"""Wraps ChromaDB's `PersistentClient` — the only place in the app that
talks to Chroma directly, so `IndexService` (and, for similarity search,
the retriever layer under `app.retrievers`) stays free of vector-store
specifics. No embeddings are computed here; `EmbeddingService` already
did that, this only persists, reads back, and searches them.
"""
from dataclasses import dataclass

import numpy as np
from chromadb.api import ClientAPI

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.models.stored_chunk import StoredChunk

MetadataValue = str | int | float | bool

logger = get_logger(__name__)

_DEFAULT_WRITE_BATCH_SIZE = 1000

# hnswlib's *filtered* kNN search (used internally whenever a `where`
# metadata filter is given) can fail to fill the requested `n_results`
# even when far more matching chunks exist and `n_results` is small —
# this is a real, non-deterministic hnswlib limitation (confirmed by
# direct reproduction against this app's own indexed data: the same
# query against the same 1,069-chunk filtered subset succeeds or fails
# depending only on which query vector is used), not a sign of
# insufficient `ef`/`M`. Raising `ef_search` does not help in practice
# either: ChromaDB's persistent HNSW segment reads `ef_search` once,
# from the segment's own metadata record, at index load time —
# `collection.modify(metadata={"hnsw:search_ef": ...})` only updates
# the collection's displayed metadata, never that segment record, so it
# has no effect on an already-built index without deleting and fully
# rebuilding the collection (a costly, disruptive, not-guaranteed-to-
# help fix left out of scope here). `query_similar_chunks` instead
# falls back to an exact (brute-force) similarity search — computed
# directly from the already-stored embeddings, no re-embedding, no
# ChromaDB write — whenever it catches exactly this failure.
_HNSW_FILTERED_SEARCH_FAILURE_SUBSTRING = "ef or m is too small"


@dataclass
class VectorMatch:
    """One similarity-search hit from `query_similar_chunks` — the raw
    material a retriever maps into a strongly typed `RetrievedChunk`.
    Carries Chroma's raw distance rather than a similarity score: turning
    a distance into a score is a retriever concern (score mapping may
    later differ per retriever), not this service's.
    """

    chunk_id: str
    chunk_text: str
    metadata: dict[str, MetadataValue]
    distance: float


class VectorStoreService:
    def __init__(
        self, client: ClientAPI, collection_name: str, write_batch_size: int = _DEFAULT_WRITE_BATCH_SIZE
    ) -> None:
        self._collection_name = collection_name
        # `embedding_function=None`: chunks are always embedded upstream by
        # EmbeddingService, so Chroma should never compute its own.
        self._collection = client.get_or_create_collection(name=collection_name, embedding_function=None)
        # Clamped against Chroma's own reported ceiling (`get_max_batch_size()`,
        # 5461 on the local Chroma version this app uses) so a
        # misconfigured `Settings.chroma_write_batch_size` can never
        # reproduce the "Batch size exceeds maximum batch size" failure
        # this batching exists to prevent — see `replace_feature_chunks`.
        self._write_batch_size = max(1, min(write_batch_size, client.get_max_batch_size()))

    @staticmethod
    def collection_exists(client: ClientAPI, collection_name: str) -> bool:
        """Checks for a collection without creating one — unlike this
        class's own constructor, which always calls `get_or_create_collection`.
        Lets read-only callers (e.g. `UploadRetriever`, looking up an
        upload session that may never have been embedded) avoid writing a
        fresh, empty collection into ChromaDB just by looking for one.
        """
        try:
            client.get_collection(name=collection_name, embedding_function=None)
            return True
        except Exception:
            return False

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def query_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, MetadataValue] | None = None,
    ) -> list[VectorMatch]:
        """Similarity search against this collection — the one place
        actual vector search happens, so every retriever goes through
        this instead of touching Chroma's query API directly.

        `n_results` is always clamped to how many chunks actually match
        `where` (never requests more than could ever be returned, and
        skips Chroma entirely — returning `[]` — when nothing matches at
        all). That clamp does *not* fully prevent Chroma's own "Cannot
        return the results in a contiguous 2D array. Probably ef or M is
        too small" failure, though: that error is hnswlib's *filtered*
        kNN search failing to fill even a small `n_results` despite far
        more matching chunks existing — confirmed non-deterministic
        against this app's real data (the identical query against the
        identical filtered subset succeeds or fails depending only on
        which query vector is used) — so it can happen even when
        `top_k` is well under the available count. If it does, this
        falls back to `_brute_force_query`: an exact similarity search
        computed directly from the already-stored embeddings, which
        cannot fail this way. See `_HNSW_FILTERED_SEARCH_FAILURE_SUBSTRING`'s
        comment for why adjusting `ef_search` isn't a viable fix here.
        """
        if top_k <= 0:
            return []

        available_chunks = self.count_matching_chunks(where)
        logger.info(
            "Vector query: feature=%r artifactType=%r available_chunks=%d requested_n_results=%d "
            "embedding_dimension=%d",
            _filter_value(where, "feature"),
            _filter_value(where, "artifactType"),
            available_chunks,
            top_k,
            len(query_embedding),
        )

        if available_chunks == 0:
            return []

        effective_n_results = min(top_k, available_chunks)

        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            if _is_hnsw_filtered_search_failure(exc):
                logger.warning(
                    "ChromaDB's filtered HNSW search failed for collection '%s' (feature=%r artifactType=%r, "
                    "%d available chunks) — falling back to exact brute-force similarity search: %s",
                    self._collection_name,
                    _filter_value(where, "feature"),
                    _filter_value(where, "artifactType"),
                    available_chunks,
                    exc,
                )
                return self._brute_force_query(query_embedding, effective_n_results, where)
            raise ExternalServiceError(
                f"ChromaDB query failed for collection '{self._collection_name}': {exc}"
            ) from exc

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            VectorMatch(chunk_id=chunk_id, chunk_text=document, metadata=metadata, distance=distance)
            for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances, strict=True)
        ]

    def _brute_force_query(
        self, query_embedding: list[float], top_k: int, where: dict[str, MetadataValue] | None
    ) -> list[VectorMatch]:
        """Exact nearest-neighbor search over every chunk matching
        `where`, computed directly from their already-stored embeddings
        — no ChromaDB ANN index involved, so it can't hit the filtered-
        search failure `query_similar_chunks` falls back from. Squared
        L2 distance, matching Chroma's own default `hnsw:space="l2"` (a
        *distance*, not a similarity — smaller is closer, exactly like
        `VectorMatch.distance` everywhere else), so callers (e.g.
        `distance_to_similarity`) see no difference between a normal and
        a brute-force result. Only used as a fallback (see
        `query_similar_chunks`); never the default path, so its O(n)
        cost per query is acceptable — n is one feature's chunk count
        for one artifact type, at most a few thousand.
        """
        try:
            result = self._collection.get(where=where, include=["documents", "metadatas", "embeddings"])
        except Exception as exc:
            raise ExternalServiceError(
                f"ChromaDB read failed for collection '{self._collection_name}': {exc}"
            ) from exc

        ids = result["ids"]
        if not ids:
            return []

        embeddings = np.asarray(result["embeddings"], dtype=np.float64)
        query = np.asarray(query_embedding, dtype=np.float64)
        distances = np.sum((embeddings - query) ** 2, axis=1)
        ranked_indices = np.argsort(distances)[:top_k]

        documents = result["documents"]
        metadatas = result["metadatas"]
        return [
            VectorMatch(
                chunk_id=ids[i], chunk_text=documents[i], metadata=metadatas[i], distance=float(distances[i])
            )
            for i in ranked_indices
        ]

    def count_matching_chunks(self, where: dict[str, MetadataValue] | None) -> int:
        """How many chunks currently match `where` — a cheap, ids-only
        read (`include=[]`: no documents/metadatas/embeddings fetched).
        Used both by `query_similar_chunks`'s own `n_results` guard and
        by callers (e.g. `RetrievalService`, for its "available
        candidates" debug diagnostics) that want this count without a
        second, differently-shaped Chroma read.
        """
        try:
            result = self._collection.get(where=where, include=[])
        except Exception as exc:
            raise ExternalServiceError(
                f"ChromaDB read failed for collection '{self._collection_name}': {exc}"
            ) from exc
        return len(result["ids"])

    def replace_feature_chunks(
        self,
        feature: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, MetadataValue]],
    ) -> None:
        """Clears every existing chunk for `feature` before writing the
        new set, so re-indexing a feature is idempotent instead of
        accumulating stale duplicates on every run.

        A large feature can produce more chunks than Chroma accepts in a
        single `collection.add()` call, so the write is split into
        sequential batches of `self._write_batch_size` — every batch
        keeps its ids/embeddings/documents/metadata aligned by slicing
        all four lists at the same positions, so nothing is dropped,
        reordered, or duplicated across batches. The same limit applies
        to deletion: `collection.delete(where=...)` resolves the filter
        to every matching id and submits all of them at once, which
        fails the exact same way `add()` does for a large feature — see
        `_delete_existing_feature_chunks`, which deletes by id in
        batches instead.

        Inputs are validated (matching lengths) *before* anything is
        deleted, so a caller bug never destroys the existing index.
        Deletion happens once, up front — Chroma has no cross-call
        transaction to wrap the delete and every add batch in, so if a
        later batch fails, the earlier batches for this write have
        already landed and the feature is left without a consistent
        index (neither the old set nor the complete new one). Rather
        than silently reporting success or leaving that ambiguous, this
        does a best-effort cleanup of whatever this call did manage to
        write (so the feature ends up empty — a known, re-indexable
        state — instead of silently partial) and always raises a clear
        error naming which batch failed and how much was written before
        it did; the caller must re-run indexing.
        """
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ExternalServiceError(
                f"ChromaDB write failed for feature '{feature}': ids ({len(ids)}), embeddings "
                f"({len(embeddings)}), documents ({len(documents)}), and metadatas ({len(metadatas)}) "
                "must all be the same length."
            )

        batches = list(_batched(ids, embeddings, documents, metadatas, self._write_batch_size))
        total_batches = len(batches)
        total_chunks = len(ids)

        self._delete_existing_feature_chunks(feature)

        written = 0
        for batch_number, (batch_ids, batch_embeddings, batch_documents, batch_metadatas) in enumerate(
            batches, start=1
        ):
            logger.info("Writing ChromaDB batch %d/%d: %d chunks", batch_number, total_batches, len(batch_ids))
            try:
                self._collection.add(
                    ids=batch_ids, embeddings=batch_embeddings, documents=batch_documents, metadatas=batch_metadatas
                )
            except Exception as exc:
                self._best_effort_delete_after_failed_write(feature)
                raise ExternalServiceError(
                    f"ChromaDB write failed for feature '{feature}' on batch {batch_number}/{total_batches}: "
                    f"{exc}. {written} of {total_chunks} chunks had already been written before this batch "
                    "failed; the feature's index has been cleared (best-effort) rather than left partially "
                    "indexed — re-run indexing for this feature."
                ) from exc
            written += len(batch_ids)

    def _delete_existing_feature_chunks(self, feature: str) -> None:
        """Deletes every chunk currently stored for `feature`, by id, in
        batches of `self._write_batch_size` — never via
        `collection.delete(where={"feature": feature})` directly, which
        resolves the filter to every matching id server-side and
        submits all of them in one call, exceeding Chroma's own batch
        ceiling for a large feature exactly like an oversized `add()`
        would. First reads back just the ids (`include=[]`: no
        documents, metadatas, or embeddings fetched — this only needs
        to know *which* chunks exist, not their content), then deletes
        them batch by batch, logging each one.

        Raises `ExternalServiceError` naming the failed batch and how
        many chunks were already deleted before it, if a delete batch
        fails — `replace_feature_chunks` never proceeds to write new
        chunks when this raises, since the exception propagates before
        the add loop is ever reached.
        """
        try:
            existing = self._collection.get(where={"feature": feature}, include=[])
        except Exception as exc:
            raise ExternalServiceError(
                f"ChromaDB read failed while preparing to delete feature '{feature}': {exc}"
            ) from exc

        existing_ids = existing["ids"]
        if not existing_ids:
            return

        id_batches = _chunked(existing_ids, self._write_batch_size)
        total_batches = len(id_batches)
        logger.info(
            "Deleting existing ChromaDB chunks for feature '%s': %d chunks in %d batches",
            feature,
            len(existing_ids),
            total_batches,
        )

        deleted = 0
        for batch_number, batch_ids in enumerate(id_batches, start=1):
            logger.info("Deleting ChromaDB batch %d/%d: %d chunks", batch_number, total_batches, len(batch_ids))
            try:
                self._collection.delete(ids=batch_ids)
            except Exception as exc:
                raise ExternalServiceError(
                    f"ChromaDB delete failed for feature '{feature}' on batch {batch_number}/{total_batches}: "
                    f"{exc}. {deleted} of {len(existing_ids)} existing chunks had already been deleted before "
                    "this batch failed; no new chunks were written for this feature — re-run indexing."
                ) from exc
            deleted += len(batch_ids)

    def _best_effort_delete_after_failed_write(self, feature: str) -> None:
        """Tries to leave `feature` with no chunks at all (rather than a
        partial write) after an add batch fails — best-effort only: if
        this also fails (e.g. Chroma itself is unreachable, or the
        cleanup delete itself fails partway through), that failure is
        swallowed so it never masks the original error, which is what
        actually explains what went wrong. Reuses
        `_delete_existing_feature_chunks` rather than a second delete
        path, so this cleanup is exactly as batch-safe as the primary
        deletion `replace_feature_chunks` performs up front.
        """
        try:
            self._delete_existing_feature_chunks(feature)
        except Exception:
            logger.warning("Best-effort cleanup after a failed ChromaDB write also failed for feature '%s'.", feature)

    def get_chunks_for_filter(self, where: dict[str, MetadataValue] | None) -> list[VectorMatch]:
        """Reads back every chunk matching `where` with no similarity
        ranking involved — the corpus a lexical (BM25) retriever needs,
        since Chroma has no native keyword search of its own. `distance`
        is always `0.0` on the returned matches (no vector comparison
        happened); callers that need a lexical relevance score compute
        their own from `chunk_text`.
        """
        try:
            result = self._collection.get(where=where, include=["documents", "metadatas"])
        except Exception as exc:
            raise ExternalServiceError(
                f"ChromaDB read failed for collection '{self._collection_name}': {exc}"
            ) from exc

        return [
            VectorMatch(chunk_id=chunk_id, chunk_text=document, metadata=metadata, distance=0.0)
            for chunk_id, document, metadata in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        ]

    def get_feature_chunks(self, feature: str) -> list[StoredChunk]:
        """Reads back every chunk currently stored for `feature` — no
        embeddings are generated or requested from OpenAI here, this only
        inspects what `replace_feature_chunks` already wrote. Embedding
        vectors are read only to report their dimension; the raw vector
        never leaves this method.
        """
        try:
            result = self._collection.get(
                where={"feature": feature}, include=["embeddings", "documents", "metadatas"]
            )
        except Exception as exc:
            raise ExternalServiceError(f"ChromaDB read failed for feature '{feature}': {exc}") from exc

        chunks: list[StoredChunk] = []
        for chunk_id, document, metadata, embedding in zip(
            result["ids"], result["documents"], result["metadatas"], result["embeddings"], strict=True
        ):
            chunks.append(
                StoredChunk(
                    chunk_id=chunk_id,
                    section_heading=metadata.get("sectionHeading"),
                    artifact_type=metadata["artifactType"],
                    source_filename=metadata["sourceFilename"],
                    word_count=len(document.split()),
                    embedding_tokens=metadata.get("embeddingTokens"),
                    embedding_dimension=len(embedding) if embedding is not None else 0,
                    embedding_exists=embedding is not None and len(embedding) > 0,
                    document_source=metadata["documentSource"],
                    page_number=metadata.get("pageNumber"),
                )
            )
        return chunks


_Batch = tuple[list[str], list[list[float]], list[str], list[dict[str, MetadataValue]]]


def _is_hnsw_filtered_search_failure(exc: Exception) -> bool:
    """Whether `exc` is specifically hnswlib's "ef or M is too small"
    failure (its exact wording has a stable typo — "contigious", not
    "contiguous" — so this matches on the more clearly-spelled half of
    the message instead of relying on that). Anything else (a real
    connection/IO failure, a malformed query) is never treated as this
    and must still surface as `ExternalServiceError` — only this one,
    specific, safely-recoverable-via-brute-force failure gets the
    fallback in `query_similar_chunks`.
    """
    return "ef or m is too small" in str(exc).lower()


def _filter_value(where: dict[str, MetadataValue] | None, key: str) -> MetadataValue | None:
    """Pulls `key`'s value back out of a Chroma `where` clause for
    logging — `where` is either a flat `{key: value}` dict or a
    `{"$and": [{k1: v1}, {k2: v2}, ...]}` combination (see
    `app.retrievers.chroma_filters.build_where_clause`, which builds
    both shapes); this handles either without callers needing to know
    which one they're holding. `None` if `where` doesn't filter on
    `key` at all (e.g. an uploaded-document query, which has no
    `artifactType` filter).
    """
    if not where:
        return None
    if key in where:
        return where[key]
    for clause in where.get("$and", []):
        if key in clause:
            return clause[key]
    return None


def _chunked(items: list, batch_size: int) -> list[list]:
    """Splits `items` into consecutive batches of at most `batch_size` —
    the one slicing rule every batched ChromaDB operation in this class
    (insert via `_batched`, delete via `_delete_existing_feature_chunks`)
    is built on, so it exists in exactly one place.
    """
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def _batched(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, MetadataValue]],
    batch_size: int,
) -> list[_Batch]:
    """Slices all four lists at the same positions, so a batch's ids,
    embeddings, documents, and metadata always stay aligned to each
    other exactly as they were in the original, unbatched input.
    Returns an empty list for empty input — the same as the caller
    simply having no chunks to write.
    """
    return list(
        zip(
            _chunked(ids, batch_size),
            _chunked(embeddings, batch_size),
            _chunked(documents, batch_size),
            _chunked(metadatas, batch_size),
            strict=True,
        )
    )
