"""Maps a raw `VectorMatch` (whatever `VectorStoreService.query_similar_chunks`
returned) into a strongly typed `RetrievedChunk` — shared by every
retriever so this metadata mapping, like `build_chunk_metadata` on the
write side, exists in exactly one place.
"""
from app.models.retrieved_chunk import RetrievedChunk
from app.retrievers.similarity import distance_to_similarity
from app.services.vector_store_service import VectorMatch


def to_retrieved_chunk(match: VectorMatch, collection_name: str) -> RetrievedChunk:
    metadata = match.metadata
    return RetrievedChunk(
        chunk_id=match.chunk_id,
        text=match.chunk_text,
        similarity_score=distance_to_similarity(match.distance),
        vector_distance=match.distance,
        artifact_type=metadata["artifactType"],
        feature=metadata["feature"],
        source_filename=metadata["sourceFilename"],
        section_heading=metadata.get("sectionHeading"),
        page_number=metadata.get("pageNumber"),
        collection_name=collection_name,
        # Absent on chunks indexed before this metadata existed — `.get()`
        # leaves them `None`, which `AdjacentChunkMerger` treats as
        # "never merge this chunk" rather than guessing at adjacency.
        document_id=metadata.get("documentId"),
        chunk_number=metadata.get("chunkNumber"),
        document_title=metadata.get("documentTitle"),
        source_path=metadata.get("sourcePath"),
        source_folder=metadata.get("sourceFolder"),
    )
