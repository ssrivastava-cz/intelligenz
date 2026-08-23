from pydantic import BaseModel


class FeatureSummary(BaseModel):
    """Document counts for one Source of Truth feature folder."""

    feature: str
    documents: int
    workflows: int
    test_cases: int
    issues: int
