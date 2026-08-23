from app.schemas.common import CamelModel


class UploadSessionResponse(CamelModel):
    session_id: str
