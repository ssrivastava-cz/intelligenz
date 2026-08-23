"""Strongly typed model for the AI's JSON response to a Knowledge
Assistant question — the exact contract enforced via OpenAI Structured
Outputs (`response_format=KnowledgeAssistantAnswer`), mirroring exactly
how `app.models.generated_test_case.GeneratedTestCasesResponse`
constrains the Test Plan Generator's response: the same approach, not a
second one. Uses `CamelModel` for the same reason that one does — the
incoming data is itself a camelCase wire format from OpenAI, not an
HTTP client.
"""
from app.schemas.common import CamelModel


class KnowledgeAssistantAnswer(CamelModel):
    """`answer` is the natural-language answer; `source_documents` must
    only ever name documents that were actually retrieved for this
    question — `KnowledgeAssistantService` independently verifies that
    after parsing (Structured Outputs constrains the *shape* of the
    response, not which document names are truthful), rejecting the
    whole response if the model names one that wasn't retrieved.
    """

    answer: str
    source_documents: list[str]
