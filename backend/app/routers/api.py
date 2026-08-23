"""Aggregates all routers under a single APIRouter mounted in app.main."""
from fastapi import APIRouter

from app.routers import (
    documents,
    feedback,
    generation,
    health,
    index_debug,
    index_service,
    indexing,
    knowledge_assistant,
    prompt,
    redmine,
    retrieval,
    source_of_truth,
    test_plans,
    upload_session,
    uploads,
    usage,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(upload_session.router)
api_router.include_router(documents.router)
api_router.include_router(uploads.router)
api_router.include_router(source_of_truth.router)
api_router.include_router(indexing.router)
api_router.include_router(index_service.router)
api_router.include_router(index_debug.router)
api_router.include_router(redmine.router)
api_router.include_router(retrieval.router)
api_router.include_router(prompt.router)
api_router.include_router(generation.router)
api_router.include_router(knowledge_assistant.router)
api_router.include_router(test_plans.router)
api_router.include_router(usage.router)
api_router.include_router(feedback.router)
