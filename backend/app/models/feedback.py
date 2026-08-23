from datetime import datetime

from pydantic import BaseModel

from app.models.common import FeedbackRating


class FeedbackEntry(BaseModel):
    id: str
    generation_id: str
    rating: FeedbackRating | None = None
    comment: str | None = None
    received_at: datetime
