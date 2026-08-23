"""FastAPI application entrypoint.

Document ingestion, Knowledge Base Indexing (real OpenAI + ChromaDB),
and Redmine ticket retrieval are real. Test Plan Generation (retrieval,
prompt building, chat completion) and MongoDB are still mock/not yet
implemented.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import get_settings
from app.core.error_handlers import register_exception_handlers
from app.core.logging import configure_logging
from app.routers.api import api_router

settings = get_settings()


def create_app() -> FastAPI:
    # `DEBUG`-level logging (e.g. the Knowledge Assistant's raw OpenAI
    # usage object — see `KnowledgeAssistantService.ask`) only ever
    # prints when `Settings.debug` is on; production stays at `INFO`.
    configure_logging(level="DEBUG" if settings.debug else "INFO", logs_root=Path(settings.logs_root))

    app = FastAPI(title=settings.app_name, debug=settings.debug)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    return app


app = create_app()
