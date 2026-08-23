"""Maps app.core.exceptions.AppError subclasses to HTTP responses.

Registered on the FastAPI app in app.main so routers/services can raise
plain application errors without knowing about HTTP status codes.
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError, ConflictError, NotFoundError, UnauthorizedError, ValidationError

_STATUS_MAP = {
    NotFoundError: 404,
    ValidationError: 422,
    UnauthorizedError: 401,
    ConflictError: 409,
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        status_code = _STATUS_MAP.get(type(exc), 500)
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})
