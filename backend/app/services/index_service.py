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
from app.models.index_history import IndexHistoryEntry
from app.services.chunk_annotator import annotate_chunks
from app.services.chunk_metadata import build_chunk_metadata
from app.services.chunking_engine import ChunkingEngine
from app.services.embedding_preview import build_embedding_preview
from app.services.embedding_service import EmbeddingService
from app.services.history_service import HistoryService
from app.services.source_of_truth_indexer import SourceOfTruthIndexer
from app.services.vector_store_service import VectorStoreService
from app.utils.datetime_utils import utcnow


class IndexService:
    def __init__(
        self,
        indexer: SourceOfTruthIndexer,
        chunking_engine: ChunkingEngine,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        history_service: HistoryService,
        embedding_model: str,
        price_per_1k_tokens: float,
    ) -> None:
        self._indexer = indexer
        self._chunking_engine = chunking_engine
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._history_service = history_service
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
        embedding_batch = self._embedding_service.embed_texts(texts)
        # Local token count per chunk (no API call — the real embedding
        # response above only reports one aggregate `total_tokens` for the
        # whole request, never a per-input breakdown) so per-chunk usage
        # can be reported and persisted.
        chunk_tokens = self._embedding_service.count_tokens(texts)

        self._vector_store.replace_feature_chunks(
            feature=feature,
            ids=[annotated.chunk.chunk_id for _, annotated in pairs],
            embeddings=embedding_batch.embeddings,
            documents=texts,
            metadatas=[
                build_chunk_metadata(item, annotated, tokens)
                for (item, annotated), tokens in zip(pairs, chunk_tokens, strict=True)
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
