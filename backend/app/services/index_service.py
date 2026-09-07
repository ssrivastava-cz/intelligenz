"""Orchestrates the Knowledge Base Indexing pipeline:

    Feature -> Discover Documents -> Parse Documents -> Generate Chunks
    -> Generate OpenAI Embeddings -> Store in ChromaDB

Every step is delegated to an existing, unmodified service
(`SourceOfTruthIndexer`, `ChunkingEngine` via `annotate_chunks`,
`EmbeddingService`, `VectorStoreService`) — this class only sequences
them and records what happened via `HistoryService`, the only component
allowed to write into `backend/data/history/`. Only Source of Truth
documents are ever indexed; there is no path here that touches user
uploads or generates a Test Plan.
"""
import time

from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.models.bm25_index import BM25ChunkRecord
from app.models.index_history import IndexAllFeatureResult, IndexAllSummary, IndexHistoryEntry
from app.retrievers.text_tokenizer import tokenize
from app.services.bm25_index_manager import BM25IndexManager
from app.services.chunk_annotator import annotate_chunks
from app.services.chunk_metadata import build_chunk_metadata
from app.services.chunking_engine import ChunkingEngine
from app.services.embedding_preview import build_embedding_preview
from app.services.embedding_service import EmbeddingService
from app.services.history_service import HistoryService
from app.services.source_of_truth_indexer import SourceOfTruthIndexer
from app.services.vector_store_service import VectorStoreService
from app.utils.datetime_utils import utcnow

logger = get_logger(__name__)


class IndexService:
    def __init__(
        self,
        indexer: SourceOfTruthIndexer,
        chunking_engine: ChunkingEngine,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        history_service: HistoryService,
        bm25_index_manager: BM25IndexManager,
        embedding_model: str,
        price_per_1k_tokens: float,
    ) -> None:
        self._indexer = indexer
        self._chunking_engine = chunking_engine
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._history_service = history_service
        self._bm25_index_manager = bm25_index_manager
        self._embedding_model = embedding_model
        self._price_per_1k_tokens = price_per_1k_tokens

    def index_feature(self, feature: str) -> IndexHistoryEntry:
        """Raises `NotFoundError` (via `SourceOfTruthIndexer.parse_feature`)
        if the feature doesn't exist. A feature with no chunkable
        documents still produces a (zero-chunk) history entry — that's
        a completed run, not an error.
        """
        started_at = time.perf_counter()

        parsed_documents = self._indexer.parse_feature(feature)
        pairs = [
            (item, annotated)
            for item in parsed_documents
            for annotated in annotate_chunks(item, self._chunking_engine)
        ]

        texts = [annotated.chunk.chunk_text for _, annotated in pairs]
        ids = [annotated.chunk.chunk_id for _, annotated in pairs]
        embedding_batch = self._embedding_service.embed_texts(texts)
        # Local token count per chunk (no API call — the real embedding
        # response above only reports one aggregate `total_tokens` for the
        # whole request, never a per-input breakdown) so per-chunk usage
        # can be reported and persisted.
        chunk_tokens = self._embedding_service.count_tokens(texts)
        metadatas = [
            build_chunk_metadata(item, annotated, tokens)
            for (item, annotated), tokens in zip(pairs, chunk_tokens, strict=True)
        ]

        self._vector_store.replace_feature_chunks(
            feature=feature,
            ids=ids,
            embeddings=embedding_batch.embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        # Same canonical chunks, same per-feature replacement contract:
        # vector search (ChromaDB) and lexical search (persistent BM25)
        # now operate over exactly one corpus. Runs only here, never at
        # query time; a per-feature re-index drops this feature's stale
        # BM25 rows and leaves every other feature's untouched.
        self._bm25_index_manager.rebuild_feature(
            feature,
            [
                BM25ChunkRecord.from_chunk(chunk_id, text, metadata, tokenize(text))
                for chunk_id, text, metadata in zip(ids, texts, metadatas, strict=True)
            ],
        )

        elapsed_seconds = time.perf_counter() - started_at
        average_tokens_per_chunk = embedding_batch.total_tokens / len(pairs) if pairs else 0.0
        entry = IndexHistoryEntry(
            feature=feature,
            indexed_at=utcnow(),
            embedding_model=self._embedding_model,
            documents_indexed=len(parsed_documents),
            chunks_indexed=len(pairs),
            embedding_tokens=embedding_batch.total_tokens,
            average_tokens_per_chunk=average_tokens_per_chunk,
            estimated_embedding_cost=(embedding_batch.total_tokens / 1000) * self._price_per_1k_tokens,
            elapsed_seconds=elapsed_seconds,
            chroma_collection_name=self._vector_store.collection_name,
        )

        embedding_preview = build_embedding_preview(
            [annotated for _, annotated in pairs], chunk_tokens, self._price_per_1k_tokens
        )
        self._history_service.save_index_history(entry, embedding_preview)

        return entry

    def index_all_features(self) -> IndexAllSummary:
        """Runs `index_feature` for every feature folder discovered under
        `source_of_truth/` (`SourceOfTruthIndexer.list_features`), one
        after another, and returns an aggregate summary.

        Sequential, not concurrent, and deliberately so: `index_feature`
        goes through `BM25IndexManager.rebuild_feature` (a single
        process-wide lock plus a full rewrite of the shared BM25 corpus
        file whose manifest is recomputed across every feature),
        `VectorStoreService.replace_feature_chunks` (a non-transactional
        delete-then-add on one shared Chroma collection), and
        `HistoryService` (run folders named by wall-clock second) — all
        shared, mutable state where interleaving buys no real speed and
        risks a Chroma/BM25/history inconsistency. Running features one
        at a time is behaviourally identical to calling
        `POST /index-feature/{feature}` once per feature.

        A feature whose `index_feature` raises is captured as a `FAILED`
        result (with the error message) and the run continues with the
        next feature. `index_feature` already guarantees a failed run
        writes no history and best-effort-clears only its own Chroma
        chunks, and per-feature replacement isolates every other
        feature's already-consistent vector and BM25 index.
        """
        started_at = time.perf_counter()
        results: list[IndexAllFeatureResult] = []

        for feature in self._indexer.list_features():
            feature_started_at = time.perf_counter()
            try:
                entry = self.index_feature(feature)
                results.append(IndexAllFeatureResult.from_entry(entry))
            except Exception as exc:  # noqa: BLE001 - one feature's failure must not abort the batch
                logger.warning("Indexing failed for feature '%s': %s", feature, exc)
                results.append(
                    IndexAllFeatureResult.failed(
                        feature, error=str(exc), elapsed_seconds=time.perf_counter() - feature_started_at
                    )
                )

        return IndexAllSummary.from_results(results, elapsed_seconds=time.perf_counter() - started_at)

    def list_history(self) -> list[IndexHistoryEntry]:
        return self._history_service.list_index_history()

    def get_latest_history_entry(self, feature: str) -> IndexHistoryEntry:
        """The most recent completed indexing run for `feature`, for the
        `/index-debug/{feature}` inspection endpoint. Raises `NotFoundError`
        if the feature has never been indexed — distinct from the feature
        not existing under Source of Truth, which `index_feature` already
        covers. This "latest for a feature" filter is index-specific
        business logic, so it lives here rather than on `HistoryService`,
        which stays a generic save/list/load/delete store reusable for
        future retrieval and generation history.
        """
        matching = [entry for entry in self._history_service.list_index_history() if entry.feature == feature]
        if not matching:
            raise NotFoundError(f"Feature '{feature}' has never been indexed.")
        return max(matching, key=lambda entry: entry.indexed_at)
