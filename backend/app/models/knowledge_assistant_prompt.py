from datetime import datetime

from pydantic import BaseModel

from app.models.retrieved_chunk import RetrievedChunk


class KnowledgeAssistantPromptInput(BaseModel):
    """Everything `KnowledgeAssistantPromptBuilder` needs to produce one
    prompt — every field here is already-resolved data (chunks already
    retrieved by `RetrievalService`). The builder itself never fetches,
    retrieves, or parses anything; it only formats what it's given, in
    list order. Separate from `app.models.prompt_input.PromptBuilderInput`
    (the existing Test Plan Generator's input) — this has no feature,
    Redmine ticket, or test-case generation instructions at all.
    """

    user_question: str
    workflow_chunks: list[RetrievedChunk]
    historical_test_case_chunks: list[RetrievedChunk]
    historical_issue_chunks: list[RetrievedChunk]
    uploaded_document_chunks: list[RetrievedChunk]


class KnowledgeAssistantPromptSections(BaseModel):
    """Every section of the generated prompt, individually addressable —
    lets a caller compute per-section token counts without re-deriving
    section boundaries by re-parsing the final assembled `prompt_text`.
    """

    system_instructions: str
    retrieved_context: str
    user_question_section: str
    answer_instructions: str


class KnowledgeAssistantPromptResult(BaseModel):
    prompt_version: str
    generated_at: datetime
    prompt_text: str
    sections: KnowledgeAssistantPromptSections
