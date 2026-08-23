from app.schemas.common import CamelModel


class IndexFeatureRequest(CamelModel):
    feature: str
    document_ids: list[str]


class IndexFeatureResponse(CamelModel):
    index_id: str
    feature: str
    status: str
    chunks_indexed: int
    message: str
