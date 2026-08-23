from pydantic import BaseModel

from app.models.chunk import Chunk


class AnnotatedChunk(BaseModel):
    """A `Chunk` enriched with the section heading it came from and its
    position within the document — metadata `Chunk` itself doesn't carry.

    Shared by the `/source-of-truth/{feature}/chunks` debug endpoint and
    `IndexService`, which both need heading-annotated chunks.
    """

    chunk: Chunk
    section_heading: str | None
    chunk_number: int
    word_count: int
