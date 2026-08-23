"""Unit tests for build_chunk_metadata — specifically the `documentId`/
`chunkNumber` fields added so a later retrieval stage (AdjacentChunkMerger)
can detect true neighbouring chunks. Every other field this function
writes is already covered indirectly by test_index_service.py /
test_upload_embedding_service.py; these tests focus on the two new keys.
"""
from app.models.common import DocumentCategory, DocumentSource, DocumentType
from app.models.parsed_document import DocumentSection, ParsedDocument, ParsedDocumentMetadata
from app.models.parsed_feature_document import ParsedFeatureDocument
from app.services.chunk_annotator import annotate_chunks
from app.services.chunk_metadata import build_chunk_metadata
from app.services.chunking_engine import ChunkingEngine


def _make_feature_document(sections: list[DocumentSection], document_id: str = "doc-1") -> ParsedFeatureDocument:
    parsed_document = ParsedDocument(
        document_id=document_id,
        title="Doc",
        sections=sections,
        content="\n\n".join(section.content for section in sections),
        metadata=ParsedDocumentMetadata(
            feature="Appointments",
            document_type=DocumentType.MARKDOWN,
            document_source=DocumentSource.SOURCE_OF_TRUTH,
            source_filename="doc.md",
            parser_name="MarkdownParser",
            parser_version="1.0",
        ),
    )
    return ParsedFeatureDocument(
        document_name="doc.md", document_category=DocumentCategory.WORKFLOW, parsed_document=parsed_document
    )


def test_build_chunk_metadata_includes_the_chunks_document_id():
    item = _make_feature_document([DocumentSection(heading="A", content="Body text.")], document_id="doc-42")
    [annotated] = annotate_chunks(item, ChunkingEngine())

    metadata = build_chunk_metadata(item, annotated, embedding_tokens=10)

    assert metadata["documentId"] == "doc-42"


def test_build_chunk_metadata_uses_the_document_wide_running_chunk_number():
    """`annotate_chunks` re-chunks one section at a time, so the
    underlying `Chunk.chunk_number` restarts at 1 for every section —
    only `AnnotatedChunk.chunk_number` (the running count across all
    sections) is a real document-wide position, so that's what must end
    up in `chunkNumber`, not `chunk.chunk_number`.
    """
    item = _make_feature_document(
        [
            DocumentSection(heading="First", content="First body."),
            DocumentSection(heading="Second", content="Second body."),
        ]
    )
    annotated_chunks = annotate_chunks(item, ChunkingEngine())
    assert len(annotated_chunks) == 2  # one chunk per section, given short content

    first_metadata = build_chunk_metadata(item, annotated_chunks[0], embedding_tokens=5)
    second_metadata = build_chunk_metadata(item, annotated_chunks[1], embedding_tokens=5)

    assert first_metadata["chunkNumber"] == 1
    assert second_metadata["chunkNumber"] == 2
    # Confirms the two chunk_number sources actually diverge: the
    # underlying per-section Chunk.chunk_number is 1 for both sections
    # (it restarts), while AnnotatedChunk.chunk_number is 1 then 2.
    assert annotated_chunks[1].chunk.chunk_number == 1
    assert annotated_chunks[1].chunk_number == 2


def test_build_chunk_metadata_gives_every_chunk_the_same_document_id():
    item = _make_feature_document(
        [
            DocumentSection(heading="First", content="First body."),
            DocumentSection(heading="Second", content="Second body."),
        ],
        document_id="doc-shared",
    )
    annotated_chunks = annotate_chunks(item, ChunkingEngine())

    metadatas = [build_chunk_metadata(item, annotated, embedding_tokens=5) for annotated in annotated_chunks]

    assert {metadata["documentId"] for metadata in metadatas} == {"doc-shared"}
