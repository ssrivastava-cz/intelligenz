"""Application-wide exception types.

Services should raise these instead of framework-specific exceptions so
business logic stays independent of FastAPI. app.core.error_handlers
translates them into HTTP responses.
"""


class AppError(Exception):
    """Base class for all application-raised errors."""


class NotFoundError(AppError):
    """Raised when a requested resource does not exist."""


class ValidationError(AppError):
    """Raised when input fails business validation rules."""


class UnauthorizedError(AppError):
    """Raised when a caller lacks permission to perform an action."""


class ExternalServiceError(AppError):
    """Raised when a dependency (OpenAI, MongoDB, ChromaDB) fails."""


class ConflictError(AppError):
    """Raised when a resource exists but is in a state that conflicts
    with the requested operation (e.g. exporting a generation that has
    no parsed test cases)."""
