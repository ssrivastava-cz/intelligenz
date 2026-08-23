from datetime import datetime

from pydantic import BaseModel, Field


class RedmineAttachment(BaseModel):
    """One Redmine attachment that was successfully downloaded to disk.
    Parsing its contents happens later, in the Retrieval Pipeline — this
    only records where the raw file landed.
    """

    filename: str
    content_type: str
    file_size: int
    local_path: str


class RedmineTicket(BaseModel):
    """A Redmine issue, as retrieved by `RedmineService`. Independent of
    OpenAI, embeddings, ChromaDB, and the retrieval pipeline — this is
    raw Redmine data plus staged attachment files, nothing more.
    """

    ticket_id: str
    title: str
    description: str
    status: str
    feature: str
    author: str
    created_at: datetime
    updated_at: datetime
    attachments: list[RedmineAttachment] = Field(default_factory=list)
