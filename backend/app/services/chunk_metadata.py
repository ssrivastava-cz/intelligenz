"""Builds the ChromaDB metadata dict for one chunk — shared by
`IndexService` (Source of Truth) and `UploadEmbeddingService` (uploaded
documents), so this mapping exists in exactly one place regardless of
which pipeline produced the chunk.
"""
from app.models.annotated_chunk import AnnotatedChunk
from app.models.parsed_feature_document import ParsedFeatureDocument
from app.services.vector_store_service import MetadataValue


def build_chunk_metadata(
    item: ParsedFeatureDocument, annotated: AnnotatedChunk, embedding_tokens: int
) -> dict[str, MetadataValue]:
    chunk = annotated.chunk
    metadata: dict[str, MetadataValue] = {
        "chunkId": chunk.chunk_id,
        "artifactType": chunk.artifact_type,
        "feature": chunk.feature,
        "documentSource": chunk.document_source,
        "sourceFilename": chunk.source_filename,
        "parserName": item.parsed_document.metadata.parser_name,
        "parserVersion": item.parsed_document.metadata.parser_version,
        "embeddingTokens": embedding_tokens,
        # Lets a retriever's `AdjacentChunkMerger` stage detect true
        # neighbours later. `annotated.chunk_number` (not `chunk.chunk_number`)
        # is used deliberately: `chunk_annotator.annotate_chunks` chunks one
        # section at a time, so `chunk.chunk_number` only counts within its
        # own section and restarts at 1 for the next one, while
        # `annotated.chunk_number` is the running count across the whole
        # document — the only one of the two that's a real document-wide
        # position.
        "documentId": chunk.document_id,
        "chunkNumber": annotated.chunk_number,
        # Always present — `ParsedDocument.title` is a required field
        # (every parser already computes one, falling back to the
        # filename-derived title when the format has no title of its
        # own), so this is never a new/independent title algorithm.
        "documentTitle": item.parsed_document.title,
    }
    # Chroma rejects `None` metadata values outright, so headings/page
    # numbers that don't apply are simply omitted rather than stored as null.
    if annotated.section_heading is not None:
        metadata["sectionHeading"] = annotated.section_heading
    if chunk.page_number is not None:
        metadata["pageNumber"] = chunk.page_number
    # `sourcePath`/`sourceFolder` are Source of Truth-only provenance —
    # `None` for a chunk from an uploaded document, which was never
    # discovered under source_of_truth/ in the first place.
    if item.parsed_document.metadata.source_path is not None:
        metadata["sourcePath"] = item.parsed_document.metadata.source_path
    if item.parsed_document.metadata.source_folder is not None:
        metadata["sourceFolder"] = item.parsed_document.metadata.source_folder
    return metadata
