from app.schemas.common import CamelModel


class HealthResponse(CamelModel):
    status: str
    app_name: str
    environment: str
