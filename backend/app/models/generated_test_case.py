"""Strongly typed models for the AI's JSON response — the exact
contract `PromptBuilder`'s Output Format section instructs the model to
follow. These reuse `CamelModel` (normally the API schema base) because
the incoming data is itself a camelCase wire format — from OpenAI,
rather than an HTTP client — so the same alias-generator bridge already
built for API schemas applies here too, instead of re-implementing it.
Persisted generation history still stores these by field name
(snake_case), matching every other model under `app.models`, since
`model_dump()` only uses aliases when explicitly asked to.

Field values like `priority`/`testType`/`automationStatus` are kept as
plain strings rather than strict enums: they're free-text categorical
output from an LLM, and rejecting an otherwise well-formed test case
over minor wording/casing differences would be far more brittle than
useful. Schema validation here enforces structure instead — every
required field present, correctly typed, `steps`/`tags` correctly
shaped — not a fixed vocabulary for those categorical values.
"""
from app.schemas.common import CamelModel


class GeneratedTestStep(CamelModel):
    step_no: int
    action: str
    expected_result: str


class GeneratedTestCase(CamelModel):
    """`steps` stays a structured list — never flattened into one
    string — so one test case can carry multiple ordered ADO test
    steps, matching how Azure DevOps Test Plans actually structure a
    test case (see `app.services.excel_export_service`, which emits one
    row per step under a shared test-case header row).

    `test_classification` is the one genuine judgment call the model
    makes about ADO structure (Smoke vs. Regression — see
    `app.services.prompt_builder._build_test_case_generation_instructions`).
    Every field below it is an ADO Test Case field the model is never
    trusted to decide: `GenerationService` always overwrites all six
    after parsing (see `_apply_ado_fields`), regardless of what the
    model returns — a test case is never allowed to self-report as
    automated, self-assign a State/Area Path/Assignee, or invent its
    own Tags. The prompt instructs the model to return `null` for all
    of them; they're nullable here so Structured Outputs accepts that,
    and so a pre-ADO persisted `GeneratedTestCase` (missing them
    entirely) still deserializes.
    """

    requirement_id: str
    test_case_id: str
    test_case_title: str
    priority: str
    test_suite: str
    preconditions: str
    steps: list[GeneratedTestStep]
    post_conditions: str
    test_type: str
    test_classification: str = "Regression"
    work_item_type: str | None = None
    automation_status: str | None = None
    state: str | None = None
    area_path: str | None = None
    assigned_to: str | None = None
    tags: list[str] | None = None


class GeneratedTestCasesResponse(CamelModel):
    """The parsed, validated shape of the AI's JSON response as a
    whole — `{"testCases": [...], "coverageNote": ...}`. `coverage_note`
    is the model's own stated reason for returning fewer test cases
    than `PromptBuilder` requested (see `_build_test_case_generation_instructions`)
    — `None` when it generated the full requested count. Never used to
    justify returning *more* than requested; `GenerationService` still
    enforces that ceiling regardless of what this says.
    """

    test_cases: list[GeneratedTestCase]
    coverage_note: str | None = None
