from pydantic import BaseModel


class SimilarityStatistics(BaseModel):
    """Aggregate view of a set of chunks' already-computed similarity
    scores — a pure diagnostic, never used to re-rank or filter
    anything. Zero for an empty chunk list rather than raising, since
    "nothing retrieved" is a normal, expected outcome to report on.
    """

    average: float
    minimum: float
    maximum: float
