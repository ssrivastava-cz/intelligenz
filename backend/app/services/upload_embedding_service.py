"""Orchestrates the Uploaded Document Embedding pipeline — the upload
equivalent of `IndexService`, but for ad-hoc user-uploaded documents
instead of the permanent Source of Truth knowledge base.

Every step is delegated to an existing, unmodified service
(`UploadedDocumentIndexer`, `ChunkingEngine` via `annotate_chunks`,
`EmbeddingService`, `VectorStoreService`) — this class only sequences
them and records what happened via `HistoryService`. Embeddings are
stored in a dedicated, per-session ChromaDB collection
(`<prefix>_<upload_session_id>`), completely independent of the Source
of Truth collection — indexing a feature can never see or affect an
uploaded document's embeddings, and vice versa.
"""
import time

from chromadb.api import ClientAPI

from app.core.exceptions import NotFoundError
from app.models.upload_history import UploadHistoryEntry, UploadHistoryRecord
from app.services.chunk_annotator import annotate_chunks
from app.services.chunk_metadata import build_chunk_metadata
from app.services.chunking_engine import ChunkingEngine
from app.services.embedding_preview import build_embedding_preview
from app.services.embedding_service import EmbeddingService
from app.services.history_service import HistoryService
from app.services.uploaded_document_indexer import UploadedDocumentIndexer
from app.services.vector_store_service import VectorStoreService
from app.utils.datetime_utils import utcnow


class UploadEmbeddingService:
    def __init__(
        self,
        indexer: UploadedDocumentIndexer,
        chunking_engine: ChunkingEngine,
        embedding_service: EmbeddingService,
        chroma_client: ClientAPI,
        history_service: HistoryService,
        embedding_model: str,
        price_per_1k_tokens: float,
        collection_prefix: str,
    ) -> None:
        self._indexer = indexer
        self._chunking_engine = chunking_engine
        self._embedding_service = embedding_service
        self._chroma_client = chroma_client
        self._history_service = history_service
        self._embedding_model = embedding_model
        self._price_per_1k_tokens = price_per_1k_tokens
        self._collection_prefix = collection_prefix

    def embed_session(self, upload_session_id: str) -> UploadHistoryEntry:
        """Raises `NotFoundError` if the upload session was never created.
        A session with zero uploaded documents still produces a
        (zero-chunk) history entry — a completed run, not an error,
        mirroring `IndexService.index_feature`.
        """
        started_at = time.perf_counter()

        parsed_documents = self._indexer.parse_session(upload_session_id)
        pairs = [
            (item, annotated)
            for item in parsed_documents
            for annotated in annotate_chunks(item, self._chunking_engine)
        ]

        texts = [annotated.chunk.chunk_text for _, annotated in pairs]
        embedding_batch = self._embedding_service.embed_texts(texts)
        chunk_tokens = self._embedding_service.count_tokens(texts)

        vector_store = self._vector_store_for_session(upload_session_id)
        vector_store.replace_feature_chunks(
            feature=upload_session_id,
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
        entry = UploadHistoryEntry(
            upload_session_id=upload_session_id,
            uploaded_at=utcnow(),
            embedding_model=self._embedding_model,
            documents_indexed=len(parsed_documents),
            chunks_indexed=len(pairs),
            embedding_tokens=embedding_batch.total_tokens,
            average_tokens_per_chunk=average_tokens_per_chunk,
            estimated_embedding_cost=(embedding_batch.total_tokens / 1000) * self._price_per_1k_tokens,
            elapsed_seconds=elapsed_seconds,
            chroma_collection_name=vector_store.collection_name,
        )

        embedding_preview = build_embedding_preview(
            [annotated for _, annotated in pairs], chunk_tokens, self._price_per_1k_tokens
        )
        self._history_service.save_upload_history(entry, embedding_preview)

        return entry

    def get_latest_history(self, upload_session_id: str) -> UploadHistoryRecord:
        """The most recent embedding run's full record (summary plus
        embedding preview) for `upload_session_id`, for the
        `/uploads/{upload_session_id}/debug` inspection endpoint. Raises
        `NotFoundError` if that session has never been embedded. This
        "latest for a session" filter is upload-specific business logic,
        so it lives here rather than on `HistoryService`, which stays a
        generic save/list store reusable for future retrieval and
        generation history — same reasoning as
        `IndexService.get_latest_history_entry`.
        """
        matching = [
            record
            for record in self._history_service.list_upload_history()
            if record.summary.upload_session_id == upload_session_id
        ]
        if not matching:
            raise NotFoundError(f"No embedding history found for upload session '{upload_session_id}'.")
        return max(matching, key=lambda record: record.summary.uploaded_at)

    def _vector_store_for_session(self, upload_session_id: str) -> VectorStoreService:
        collection_name = f"{self._collection_prefix}_{upload_session_id}"
        return VectorStoreService(client=self._chroma_client, collection_name=collection_name)
