"""Stores feedback submitted for a generation. Mock only — in-memory."""
from app.models.common import FeedbackRating
from app.models.feedback import FeedbackEntry
from app.utils.datetime_utils import utcnow
from app.utils.ids import generate_id


class FeedbackService:
    def __init__(self) -> None:
        self._entries: list[FeedbackEntry] = []

    def submit_feedback(
        self,
        generation_id: str,
        rating: FeedbackRating | None,
        comment: str | None,
    ) -> FeedbackEntry:
        entry = FeedbackEntry(
            id=generate_id("fb"),
            generation_id=generation_id,
            rating=rating,
            comment=comment,
            received_at=utcnow(),
        )
        self._entries.append(entry)
        return entry
