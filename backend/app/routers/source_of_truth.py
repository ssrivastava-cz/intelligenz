from fastapi import APIRouter

from app.config.settings import get_settings
from app.core.dependencies import ChunkingEngineDep, EmbeddingServiceDep, SourceOfTruthIndexerDep
from app.models.annotated_chunk import AnnotatedChunk
from app.schemas.source_of_truth import (
    ChunkOut,
    EmbeddingPreviewChunkOut,
    EmbeddingPreviewResponse,
    FeatureSummaryOut,
    ParsedFeatureDocumentOut,
    SourceOfTruthParsedResponse,
)
from app.services.chunk_annotator import annotate_chunks
from app.services.embedding_preview import build_embedding_preview

router = APIRouter(tags=["source-of-truth"])


@router.get("/source-of-truth", response_model=list[FeatureSummaryOut])
async def list_source_of_truth_features(indexer: SourceOfTruthIndexerDep) -> list[FeatureSummaryOut]:
    return [FeatureSummaryOut.model_validate(summary) for summary in indexer.summarize_all_features()]


@router.get("/source-of-truth/{feature}/parsed", response_model=SourceOfTruthParsedResponse)
async def get_parsed_source_of_truth_feature(
    feature: str,
    indexer: SourceOfTruthIndexerDep,
) -> SourceOfTruthParsedResponse:
    """Scans, parses, and returns every document for a feature — for
    inspecting parser output before chunking. No chunks, no embeddings.
    """
    parsed_documents = indexer.parse_feature(feature)
    return SourceOfTruthParsedResponse(
        feature=feature,
        documents=[ParsedFeatureDocumentOut.from_parsed(item) for item in parsed_documents],
    )


@router.get("/source-of-truth/{feature}/chunks", response_model=list[ChunkOut])
async def get_source_of_truth_feature_chunks(
    feature: str,
    indexer: SourceOfTruthIndexerDep,
    chunking_engine: ChunkingEngineDep,
) -> list[ChunkOut]:
    """Chunks every Source of Truth document for a feature — a developer
    debugging tool to verify `ChunkingEngine` output before the Knowledge
    Base Indexing subsystem exists. No embeddings, no ChromaDB.
    """
    parsed_documents = indexer.parse_feature(feature)
    chunks: list[ChunkOut] = []
    for item in parsed_documents:
        chunks.extend(_to_chunk_out(annotated) for annotated in annotate_chunks(item, chunking_engine))
    return chunks


def _to_chunk_out(annotated: AnnotatedChunk) -> ChunkOut:
    return ChunkOut(
        chunk_id=annotated.chunk.chunk_id,
        artifact_type=annotated.chunk.artifact_type,
        feature=annotated.chunk.feature,
        document_source=annotated.chunk.document_source,
        source_filename=annotated.chunk.source_filename,
        section_heading=annotated.section_heading,
        page_number=annotated.chunk.page_number,
        chunk_number=annotated.chunk_number,
        word_count=annotated.word_count,
        chunk_text=annotated.chunk.chunk_text,
    )


@router.get("/source-of-truth/{feature}/embedding-preview", response_model=EmbeddingPreviewResponse)
async def get_source_of_truth_embedding_preview(
    feature: str,
    indexer: SourceOfTruthIndexerDep,
    chunking_engine: ChunkingEngineDep,
    embedding_service: EmbeddingServiceDep,
) -> EmbeddingPreviewResponse:
    """Previews embedding token usage and cost for a feature before it's
    actually indexed — a developer debugging tool. Tokens are counted
    locally via `EmbeddingService.count_tokens` (tiktoken); no OpenAI
    call is made, no embeddings are generated, and nothing is written to
    ChromaDB or the indexing history.
    """
    parsed_documents = indexer.parse_feature(feature)
    pairs = [
        (item, annotated)
        for item in parsed_documents
        for annotated in annotate_chunks(item, chunking_engine)
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

    return EmbeddingPreviewResponse(
        feature=feature,
        embedding_model=embedding_service.model,
        documents_found=len(parsed_documents),
        chunks_created=chunks_created,
        total_embedding_tokens=total_tokens,
        average_tokens_per_chunk=average_tokens_per_chunk,
        estimated_embedding_cost=(total_tokens / 1000) * price_per_1k_tokens,
        chunks=[EmbeddingPreviewChunkOut.model_validate(chunk) for chunk in preview_chunks],
    )
