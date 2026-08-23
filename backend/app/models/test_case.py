from pydantic import BaseModel

from app.models.common import Priority, TestCaseSource, TestType


class TestCase(BaseModel):
    id: str
    feature: str
    test_case: str
    preconditions: str
    steps: list[str]
    expected_result: str
    priority: Priority
    type: TestType
    source: TestCaseSource
    automation_candidate: bool
