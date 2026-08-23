from fastapi import APIRouter

from app.core.dependencies import IndexServiceDep, VectorStoreServiceDep
from app.schemas.index_debug import IndexDebugChunkOut, IndexDebugResponse

router = APIRouter(tags=["knowledge-base-indexing"])


@router.get("/index-debug/{feature}", response_model=IndexDebugResponse)
async def get_index_debug(
    feature: str,
    index_service: IndexServiceDep,
    vector_store: VectorStoreServiceDep,
) -> IndexDebugResponse:
    """Read-only inspection of what's already indexed for a feature: the
    last recorded indexing run plus every chunk currently stored in
    ChromaDB. Generates no embeddings and makes no OpenAI calls — it only
    reads what `IndexService.index_feature` already wrote.
    """
    latest_entry = index_service.get_latest_history_entry(feature)
    stored_chunks = vector_store.get_feature_chunks(feature)

    return IndexDebugResponse(
        feature=latest_entry.feature,
        embedding_model=latest_entry.embedding_model,
        documents_indexed=latest_entry.documents_indexed,
        chunks_indexed=latest_entry.chunks_indexed,
        embedding_tokens=latest_entry.embedding_tokens,
        estimated_cost=latest_entry.estimated_embedding_cost,
        average_tokens_per_chunk=latest_entry.average_tokens_per_chunk,
        elapsed_time_seconds=latest_entry.elapsed_seconds,
        indexed_at=latest_entry.indexed_at,
        collection=vector_store.collection_name,
        chunks=[IndexDebugChunkOut.model_validate(chunk) for chunk in stored_chunks],
    )
