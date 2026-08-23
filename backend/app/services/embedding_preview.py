"""Builds per-chunk embedding-cost previews — shared by the
`/source-of-truth/{feature}/embedding-preview` debug endpoint and
`IndexService` (which persists the same preview via `HistoryService`
right before generating real embeddings), so this arithmetic exists in
exactly one place.
"""
from app.models.annotated_chunk import AnnotatedChunk
from app.models.embedding_preview import EmbeddingPreviewChunk


def build_embedding_preview(
    annotated_chunks: list[AnnotatedChunk],
    chunk_tokens: list[int],
    price_per_1k_tokens: float,
) -> list[EmbeddingPreviewChunk]:
    return [
        EmbeddingPreviewChunk(
            chunk_id=annotated.chunk.chunk_id,
            section_heading=annotated.section_heading,
            artifact_type=annotated.chunk.artifact_type,
            source_filename=annotated.chunk.source_filename,
            page_number=annotated.chunk.page_number,
            word_count=annotated.word_count,
            embedding_tokens=tokens,
            estimated_cost=(tokens / 1000) * price_per_1k_tokens,
            chunk_text=annotated.chunk.chunk_text,
        )
        for annotated, tokens in zip(annotated_chunks, chunk_tokens, strict=True)
    ]
