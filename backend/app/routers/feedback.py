from fastapi import APIRouter

from app.core.dependencies import FeedbackServiceDep
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(
    payload: FeedbackRequest,
    service: FeedbackServiceDep,
) -> FeedbackResponse:
    entry = service.submit_feedback(payload.generation_id, payload.rating, payload.comment)
    return FeedbackResponse(
        feedback_id=entry.id,
        generation_id=entry.generation_id,
        received_at=entry.received_at,
        message="Thanks for your feedback!",
    )
