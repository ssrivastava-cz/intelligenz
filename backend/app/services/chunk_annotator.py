"""Pairs each `Chunk` `ChunkingEngine` produces with the section heading
it came from, since `Chunk` itself carries no such field.

Shared by the `/source-of-truth/{feature}/chunks` debug endpoint and
`IndexService` — both need heading-annotated chunks, so the pairing
logic lives here once instead of being duplicated in each caller.
"""
from app.models.annotated_chunk import AnnotatedChunk
from app.models.parsed_feature_document import ParsedFeatureDocument
from app.services.chunking_engine import ChunkingEngine


def annotate_chunks(item: ParsedFeatureDocument, chunking_engine: ChunkingEngine) -> list[AnnotatedChunk]:
    """Chunks one section at a time so each resulting `Chunk` can be
    paired with its section heading. Chunking a single-section copy of
    the document produces exactly what `chunk_document` would produce
    for that section within the full document, since sections are
    chunked independently of one another — no chunking logic is
    duplicated here, `ChunkingEngine.chunk_document` is still the only
    thing doing the actual splitting.
    """
    entries: list[AnnotatedChunk] = []
    chunk_number = 0
    for section in item.parsed_document.sections:
        section_only_document = item.parsed_document.model_copy(update={"sections": [section]})
        for chunk in chunking_engine.chunk_document(section_only_document, item.document_category):
            chunk_number += 1
            entries.append(
                AnnotatedChunk(
                    chunk=chunk,
                    section_heading=section.heading,
                    chunk_number=chunk_number,
                    word_count=len(chunk.chunk_text.split()),
                )
            )
    return entries
