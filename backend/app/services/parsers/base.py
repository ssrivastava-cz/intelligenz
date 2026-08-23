"""Common interface every format-specific document parser implements."""
from abc import ABC, abstractmethod

from app.models.document import Document
from app.models.parsed_document import ParsedDocument


class DocumentParser(ABC):
    """Parses raw file bytes into a normalized `ParsedDocument`.

    Implementations must stay independent of OpenAI, embeddings,
    ChromaDB, and RAG — this subsystem only extracts structured text,
    nothing more. A parser never touches the filesystem itself: it's
    handed the bytes to parse plus the `Document` ingestion record
    (feature/source/filename/id) so it doesn't need to re-derive that
    metadata. This also keeps parsers trivially unit-testable — no
    disk I/O required to exercise one.

    The same interface, and the same `ParsedDocument` shape, applies
    whether `document.source` is Source of Truth or a user upload —
    parsing doesn't care which.
    """

    @abstractmethod
    def parse(self, document: Document, content: bytes) -> ParsedDocument:
        raise NotImplementedError
