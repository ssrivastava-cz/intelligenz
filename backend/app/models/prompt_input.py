from pydantic import BaseModel

from app.models.retrieved_chunk import RetrievedChunk


class PromptBuilderInput(BaseModel):
    """Everything `PromptBuilder` needs to produce one prompt — every
    field here is already-resolved data (chunks already retrieved, a
    ticket description already fetched elsewhere). `PromptBuilder` itself
    never fetches, retrieves, or parses anything; it only formats what
    it's given, in list order (no re-sorting or re-filtering).
    """

    feature: str
    redmine_ticket: str
    redmine_description: str
    user_description: str | None = None
    workflow_chunks: list[RetrievedChunk]
    historical_test_case_chunks: list[RetrievedChunk]
    historical_issue_chunks: list[RetrievedChunk]
    uploaded_document_chunks: list[RetrievedChunk]
