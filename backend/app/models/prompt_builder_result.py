from datetime import datetime

from pydantic import BaseModel


class PromptSections(BaseModel):
    """Every section of the generated prompt, individually addressable —
    lets a caller (the `/prompt/debug` endpoint) compute per-section
    token counts without re-deriving section boundaries by re-parsing
    the final assembled `prompt_text` string.
    """

    header: str
    system_instructions: str
    feature_section: str
    redmine_description_section: str
    uploaded_documents_section: str
    workflow_section: str
    historical_test_cases_section: str
    historical_issues_section: str
    test_case_generation_instructions: str
    generation_guidelines: str
    output_format: str


class PromptBuilderResult(BaseModel):
    prompt_version: str
    generated_at: datetime
    prompt_text: str
    sections: PromptSections
