from datetime import datetime

from app.models.common import FeedbackRating
from app.schemas.common import CamelModel


class FeedbackRequest(CamelModel):
    generation_id: str
    rating: FeedbackRating | None = None
    comment: str | None = None


class FeedbackResponse(CamelModel):
    feedback_id: str
    generation_id: str
    received_at: datetime
    message: str
