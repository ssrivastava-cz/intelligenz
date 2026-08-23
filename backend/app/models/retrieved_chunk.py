from pydantic import BaseModel

from app.models.common import DocumentCategory


class RetrievedChunk(BaseModel):
    """One chunk returned by a `Retriever` — the strongly typed shape
    every retriever (`SourceOfTruthRetriever`, `UploadRetriever`, and
    later `RedmineRetriever`) returns, regardless of which ChromaDB
    collection it actually came from. This is what `RetrievalService`
    merges, deduplicates, and sorts, and what the future Prompt Builder
    will consume without needing to know where a chunk originated.
    """

    chunk_id: str
    text: str
    similarity_score: float
    artifact_type: DocumentCategory
    feature: str
    source_filename: str
    section_heading: str | None
    page_number: int | None
    collection_name: str
    # The source document this chunk came from, and its position within
    # it — used by `AdjacentChunkMerger` to detect true neighbours.
    # `None` for chunks indexed before this metadata existed (older
    # ChromaDB collections never wrote it), in which case the chunk is
    # simply never merged with anything, rather than guessed at.
    document_id: str | None = None
    chunk_number: int | None = None
    # Chroma's raw distance behind `similarity_score` — kept alongside
    # it (not just derivable by inverting `distance_to_similarity`) so a
    # debug view can show exactly what the vector search returned,
    # without recomputing anything. Defaulted so existing test fixtures
    # that don't care about it still construct a `RetrievedChunk` fine.
    vector_distance: float = 0.0
    # Set only by `AdjacentChunkMerger` when this chunk is the result of
    # combining a run of originals — the sorted `chunk_number`s that
    # went into it, so a debug view can report "Merged: Yes, Range
    # 18-20, Count 3". `None` means this chunk was never merged.
    merged_chunk_numbers: list[int] | None = None
