from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from app.config.settings import get_settings
from app.core.dependencies import (
    ChunkingEngineDep,
    CostCalculatorDep,
    EmbeddingServiceDep,
    UploadedDocumentIndexerDep,
    UploadEmbeddingServiceDep,
    UploadServiceDep,
)
from app.schemas.documents import DocumentOut
from app.schemas.source_of_truth import EmbeddingPreviewChunkOut
from app.schemas.uploads import (
    UploadDebugResponse,
    UploadEmbeddingPreviewChunkOut,
    UploadEmbeddingPreviewResponse,
    UploadEmbeddingResponse,
    UploadFilesResponse,
)
from app.services.chunk_annotator import annotate_chunks
from app.services.embedding_preview import build_embedding_preview

router = APIRouter(tags=["uploads"])


@router.post("/uploads", response_model=UploadFilesResponse)
async def upload_files(
    upload_service: UploadServiceDep,
    files: Annotated[list[UploadFile], File(...)],
) -> UploadFilesResponse:
    """Creates a new upload session and persists the uploaded files —
    nothing else. No parsing, no chunking, no token counting, no
    embeddings, no ChromaDB, no history. Every later step
    (`/embedding-preview`, `/embed`, `/debug`) operates on the session
    id this returns.
    """
    session = upload_service.create_session()
    documents = await upload_service.upload_user_documents(session_id=session.id, feature=session.id, files=files)

    return UploadFilesResponse(
        upload_session_id=session.id,
        documents_uploaded=len(documents),
        uploaded_files=[DocumentOut.from_document(document) for document in documents],
    )


@router.post("/uploads/{upload_session_id}/embedding-preview", response_model=UploadEmbeddingPreviewResponse)
async def preview_upload_embeddings(
    upload_session_id: str,
    upload_indexer: UploadedDocumentIndexerDep,
    chunking_engine: ChunkingEngineDep,
    embedding_service: EmbeddingServiceDep,
    cost_calculator: CostCalculatorDep,
) -> UploadEmbeddingPreviewResponse:
    """Previews embedding token usage and cost for an already-uploaded
    session's documents — exactly like
    `/source-of-truth/{feature}/embedding-preview`, except the source is
    an upload session instead of a Source of Truth feature. Reads files
    already on disk (via `/uploads`); no new files are accepted here.
    Tokens are counted locally via `EmbeddingService.count_tokens`
    (tiktoken). No OpenAI call, no ChromaDB write, no history.
    """
    parsed_documents = upload_indexer.parse_session(upload_session_id)
    pairs = [
        (item, annotated) for item in parsed_documents for annotated in annotate_chunks(item, chunking_engine)
    ]

    annotated_chunks = [annotated for _, annotated in pairs]
    chunk_texts = [annotated.chunk.chunk_text for annotated in annotated_chunks]
    chunk_tokens = embedding_service.count_tokens(chunk_texts)

    settings = get_settings()
    price_per_1k_tokens = settings.embedding_price_per_1k_tokens
    total_tokens = sum(chunk_tokens)
    chunks_created = len(pairs)
    average_tokens_per_chunk = total_tokens / chunks_created if chunks_created else 0.0
    preview_chunks = build_embedding_preview(annotated_chunks, chunk_tokens, price_per_1k_tokens)
    estimated_embedding_cost = (total_tokens / 1000) * price_per_1k_tokens

    return UploadEmbeddingPreviewResponse(
        upload_session_id=upload_session_id,
        documents_found=len(parsed_documents),
        chunks_created=chunks_created,
        embedding_model=embedding_service.model,
        total_embedding_tokens=total_tokens,
        average_tokens_per_chunk=average_tokens_per_chunk,
        estimated_embedding_cost=estimated_embedding_cost,
        estimated_embedding_cost_inr=cost_calculator.convert_to_money(estimated_embedding_cost).inr,
        chunks=[UploadEmbeddingPreviewChunkOut.from_preview_chunk(chunk) for chunk in preview_chunks],
    )


@router.post("/uploads/{upload_session_id}/embed", response_model=UploadEmbeddingResponse)
async def embed_uploaded_documents(
    upload_session_id: str, upload_embedding_service: UploadEmbeddingServiceDep
) -> UploadEmbeddingResponse:
    """Runs the full Uploaded Document Embedding pipeline for an existing
    upload session: parse -> chunk -> embed (OpenAI) -> store (a
    dedicated ChromaDB collection for this session) -> record history.
    No upload happens here — the session's files must already exist on
    disk (via `/uploads`). Completely independent of the Source of Truth
    knowledge base.
    """
    entry = upload_embedding_service.embed_session(upload_session_id)
    return UploadEmbeddingResponse.model_validate(entry)


@router.get("/uploads/{upload_session_id}/debug", response_model=UploadDebugResponse)
async def get_upload_debug(
    upload_session_id: str, upload_embedding_service: UploadEmbeddingServiceDep
) -> UploadDebugResponse:
    """Read-only inspection of what's already been embedded for an
    upload session: the last recorded embedding run's summary plus its
    persisted embedding preview. Never regenerates embeddings, never
    calls OpenAI, never touches ChromaDB or history — only reads what
    `embed_session` already wrote to disk.
    """
    record = upload_embedding_service.get_latest_history(upload_session_id)
    return UploadDebugResponse(
        upload_session_id=record.summary.upload_session_id,
        uploaded_at=record.summary.uploaded_at,
        embedding_model=record.summary.embedding_model,
        documents_indexed=record.summary.documents_indexed,
        chunks_indexed=record.summary.chunks_indexed,
        embedding_tokens=record.summary.embedding_tokens,
        average_tokens_per_chunk=record.summary.average_tokens_per_chunk,
        estimated_embedding_cost=record.summary.estimated_embedding_cost,
        elapsed_seconds=record.summary.elapsed_seconds,
        chroma_collection_name=record.summary.chroma_collection_name,
        index_status=record.summary.index_status,
        chunks=[EmbeddingPreviewChunkOut.model_validate(chunk) for chunk in record.embedding_preview],
    )
