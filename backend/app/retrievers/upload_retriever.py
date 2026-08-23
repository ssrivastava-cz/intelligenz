"""Retrieves uploaded document chunks from an upload session's dedicated
ChromaDB collection. `RetrievalService` only ever hands this a session
id — resolving that into an actual collection name (and checking
whether one even exists yet, without ever creating it) is this
retriever's job, mirroring the `<prefix>_<upload_session_id>` scheme
`UploadEmbeddingService` already established on the write side.
"""
from chromadb.api import ClientAPI

from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.base import Retriever
from app.retrievers.chroma_filters import build_where_clause
from app.retrievers.chunk_mapping import to_retrieved_chunk
from app.services.vector_store_service import VectorStoreService


class UploadRetriever(Retriever):
    def __init__(self, chroma_client: ClientAPI, collection_prefix: str, upload_session_id: str) -> None:
        self._chroma_client = chroma_client
        self._collection_prefix = collection_prefix
        self._upload_session_id = upload_session_id

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        collection_name = f"{self._collection_prefix}_{self._upload_session_id}"
        if not VectorStoreService.collection_exists(self._chroma_client, collection_name):
            # Never embedded (or never uploaded to) yet — an empty result,
            # not an error, and critically: this read-only lookup never
            # creates the collection just by checking for it.
            return []

        vector_store = VectorStoreService(client=self._chroma_client, collection_name=collection_name)
        matches = vector_store.query_similar_chunks(query_embedding, top_k, where=build_where_clause(filters))
        return [to_retrieved_chunk(match, collection_name) for match in matches]

    def count_available(self, filters: dict[str, str] | None = None) -> int:
        """How many chunks currently match `filters` in this upload
        session's collection — `0` (not an error) if the session was
        never embedded, exactly like `retrieve()`, and never creates the
        collection just by checking.
        """
        collection_name = f"{self._collection_prefix}_{self._upload_session_id}"
        if not VectorStoreService.collection_exists(self._chroma_client, collection_name):
            return 0
        vector_store = VectorStoreService(client=self._chroma_client, collection_name=collection_name)
        return vector_store.count_matching_chunks(where=build_where_clause(filters))
