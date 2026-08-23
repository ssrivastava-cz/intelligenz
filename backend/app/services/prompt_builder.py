"""Pure formatting component: turns already-retrieved, already-resolved
structured data into one deterministic prompt string. Never calls
OpenAI, never queries ChromaDB, never retrieves documents, never parses
files, never talks to Redmine directly — every input it needs is handed
to it already resolved by whatever orchestrates it (today, `GET
/prompt/debug`; eventually, the Test Plan Generator).
"""
from datetime import datetime

from app.models.prompt_builder_result import PromptBuilderResult, PromptSections
from app.models.prompt_input import PromptBuilderInput
from app.models.retrieved_chunk import RetrievedChunk

PROMPT_VERSION = "1.0.0"

_DEFAULT_MAX_TEST_CASES = 5


class PromptBuilder:
    def __init__(self, max_test_cases: int = _DEFAULT_MAX_TEST_CASES) -> None:
        """`max_test_cases` is fixed at construction (from configuration,
        via dependency injection — see `get_prompt_builder`), never per
        call, so `build()` stays a pure function of `prompt_input` and
        `generated_at` alone, just like before.
        """
        self._max_test_cases = max_test_cases

    def build(self, prompt_input: PromptBuilderInput, generated_at: datetime) -> PromptBuilderResult:
        """`generated_at` is a parameter rather than something this method
        reads off a clock itself, so the same input always produces the
        exact same output — this stays a pure function of its arguments.
        """
        sections = PromptSections(
            header=_build_header(generated_at),
            system_instructions=_build_system_instructions(),
            feature_section=_build_feature_section(prompt_input),
            redmine_description_section=_build_redmine_description_section(prompt_input.redmine_description),
            uploaded_documents_section=_build_chunk_section(
                "Uploaded Documents", prompt_input.uploaded_document_chunks, "uploaded documents"
            ),
            workflow_section=_build_chunk_section(
                "Workflow Context", prompt_input.workflow_chunks, "workflow sections"
            ),
            historical_test_cases_section=_build_chunk_section(
                "Historical Test Cases", prompt_input.historical_test_case_chunks, "historical test cases"
            ),
            historical_issues_section=_build_chunk_section(
                "Historical Issues",
                prompt_input.historical_issue_chunks,
                "historical issues",
                intro=(
                    "The following represent previous defects for this feature. "
                    "Consider them when generating regression coverage."
                ),
            ),
            test_case_generation_instructions=_build_test_case_generation_instructions(self._max_test_cases),
            generation_guidelines=_GENERATION_GUIDELINES,
            output_format=_OUTPUT_FORMAT,
        )

        prompt_text = "\n\n".join(
            [
                sections.header,
                sections.system_instructions,
                sections.feature_section,
                sections.redmine_description_section,
                sections.uploaded_documents_section,
                sections.workflow_section,
                sections.historical_test_cases_section,
                sections.historical_issues_section,
                sections.test_case_generation_instructions,
                sections.generation_guidelines,
                sections.output_format,
            ]
        )

        return PromptBuilderResult(
            prompt_version=PROMPT_VERSION,
            generated_at=generated_at,
            prompt_text=prompt_text,
            sections=sections,
        )


def _build_header(generated_at: datetime) -> str:
    return f"Prompt Version: {PROMPT_VERSION}\nGenerated At: {generated_at.isoformat()}"


def _build_system_instructions() -> str:
    return (
        "## System Instructions\n\n"
        "You are a Senior QA Automation Engineer.\n"
        "Generate comprehensive Azure DevOps (ADO) ready manual test cases.\n"
        "Use the retrieved knowledge base as the primary source of truth.\n"
        "Prefer Workflow Documents over assumptions.\n"
        "Use Historical Test Cases to maintain consistency.\n"
        "Use Historical Issues to prevent regressions.\n"
        "Use Uploaded Documents as additional implementation context.\n\n"
        "If information conflicts, prioritize in this order:\n"
        "1. Uploaded Documents\n"
        "2. Redmine Description\n"
        "3. Workflow Documents\n"
        "4. Historical Test Cases\n"
        "5. Historical Issues\n\n"
        "Never invent functionality not supported by the provided context."
    )


def _build_feature_section(prompt_input: PromptBuilderInput) -> str:
    lines = [
        "## Feature",
        "",
        f"Feature Name: {prompt_input.feature}",
    ]
    # The Redmine ticket is optional — generation can run from the
    # Feature alone, in which case there's no ticket reference to show.
    if prompt_input.redmine_ticket:
        lines.append(f"Redmine Ticket: {prompt_input.redmine_ticket}")
    if prompt_input.user_description:
        lines.append(f"User Description: {prompt_input.user_description}")
    return "\n".join(lines)


def _build_redmine_description_section(redmine_description: str) -> str:
    if not redmine_description:
        return "## Redmine Description\n\n_No Redmine ticket description was provided for this generation._"
    return f"## Redmine Description\n\n{redmine_description}"


def _build_chunk_section(
    title: str,
    chunks: list[RetrievedChunk],
    empty_label: str,
    intro: str | None = None,
) -> str:
    """Shared by every "list of retrieved chunks" section (Uploaded
    Documents, Workflow Context, Historical Test Cases, Historical
    Issues) so this formatting exists in exactly one place. Chunks are
    rendered in the order given — never re-sorted — preserving whatever
    order the Retrieval Pipeline returned, along with each chunk's
    section heading and source document name.
    """
    blocks = [f"## {title}", ""]
    if intro:
        blocks.extend([intro, ""])

    if not chunks:
        blocks.append(f"_No {empty_label} were retrieved for this feature._")
        return "\n".join(blocks)

    for chunk in chunks:
        blocks.append(f"### {chunk.source_filename}")
        if chunk.section_heading:
            blocks.append(f"Section: {chunk.section_heading}")
        blocks.append(chunk.text)
        blocks.append("")
    return "\n".join(blocks).rstrip()


def _build_test_case_generation_instructions(max_test_cases: int) -> str:
    return (
        "## Test Case Generation Instructions\n\n"
        "Generate Azure DevOps (ADO) style test cases, structured so they can be imported into ADO Test Plans "
        "directly. "
        f"Generate exactly {max_test_cases} test cases. "
        f"Generate that many whenever the retrieved context supports {max_test_cases} genuinely distinct, "
        "source-grounded test scenarios (positive, negative, boundary, business rule, UI, regression, and "
        "integration scenarios all count toward this). "
        f"Never generate more than {max_test_cases} test cases. "
        "Do not reach the target count by duplicating a test case, creating near-identical variants of the "
        "same scenario, or fabricating a scenario the retrieved context does not support. "
        "If the retrieved context only supports fewer genuinely distinct scenarios than that, generate that "
        "smaller number instead, and set \"coverageNote\" to a short, specific explanation of what additional "
        "evidence would be needed to reach the full count. Leave \"coverageNote\" null when the full requested "
        "count was generated.\n\n"
        "## ADO Test Case Structure\n\n"
        "Each test case represents exactly one test scenario, with exactly one title, and can contain "
        "multiple ordered steps. Do NOT create a separate test case for every step in a workflow — if several "
        "actions belong to the same scenario (for example \"Navigate to a patient\", then \"Click the Pencil "
        "icon\", then \"Search in the code field\"), those are steps 1, 2, 3 of ONE test case, never three "
        "separate test cases.\n\n"
        "Number steps sequentially starting at 1 within each test case. Step numbering always restarts at 1 "
        "for the next test case — never continues counting from the previous test case's last step.\n\n"
        "Every step's \"action\" must be a concrete, executable user or system action — never a vague "
        "instruction like \"Verify functionality\". For example, write \"Click the Delete option from the "
        "kebab menu\", not \"Delete the record\" or \"Verify deletion works\".\n\n"
        "Every step's \"expectedResult\" must describe what should happen immediately after that specific "
        "action — not a summary of the whole scenario.\n\n"
        "Keep test cases independent and meaningful: each one must be executable and verifiable on its own, "
        "without depending on another generated test case having just run.\n\n"
        "## Test Case Classification\n\n"
        "Set \"testClassification\" to exactly \"Smoke\" or \"Regression\" on every test case — never any "
        "other value, and a test case is always exactly one of the two, never both.\n\n"
        "Use \"Smoke\" for: critical happy-path functionality; core functionality that should be verified "
        "during daily/basic verification; essential functionality required to confirm the feature works at "
        "all.\n\n"
        "Use \"Regression\" for: edge cases; negative scenarios; previously fixed defects; detailed "
        "validation; complex business rules; any scenario that is not part of basic smoke coverage.\n\n"
        "\"testClassification\" is separate from \"testType\" (the existing Functional/Regression/Smoke/"
        "Integration category) — always set both.\n\n"
        "## Fields Set By The Application, Not You\n\n"
        "Always return `null` for \"workItemType\", \"automationStatus\", \"state\", \"areaPath\", "
        "\"assignedTo\", and \"tags\" on every test case. The application assigns every one of these after "
        "generation — Work Item Type is always \"Test Case\", Automation Status is always \"Not Automated\", "
        "and State is always \"Design\" for a newly generated test case; Area Path, Assigned To, and Tags "
        "come from the selected feature and the application's own existing configuration, never from "
        "anything you invent. Do not guess a value for any of them — `null` is always correct.\n\n"
        "Each generated test case must include the following fields:\n\n"
        "| Column | Description |\n"
        "| --- | --- |\n"
        "| Requirement ID | User Story / Requirement ID |\n"
        "| Test Case ID | Unique Test Case Identifier |\n"
        "| Test Case Title | One clear descriptive title per test case |\n"
        "| Priority | High / Medium / Low |\n"
        "| Test Suite | Appropriate Test Suite |\n"
        "| Preconditions | Required setup |\n"
        "| Step No. | Sequential step number, restarting at 1 per test case |\n"
        "| Action | Concrete user/system action |\n"
        "| Expected Result | Expected outcome of that specific action |\n"
        "| Post Conditions | System state after execution |\n"
        "| Test Type | Functional / Regression / Smoke / Integration |\n"
        "| Test Classification | Smoke or Regression — see above |\n"
    )


_GENERATION_GUIDELINES = (
    "## Generation Guidelines\n\n"
    "Generate:\n"
    "- Positive scenarios\n"
    "- Negative scenarios\n"
    "- Boundary cases\n"
    "- Business rule validation\n"
    "- UI validation\n"
    "- Regression coverage\n"
    "- Integration scenarios\n\n"
    "Avoid duplicate test cases. Reuse historical terminology whenever possible."
)

_OUTPUT_FORMAT = (
    "## Output Format\n\n"
    "Return only valid JSON.\n"
    "Do not return Markdown.\n"
    "Do not return tables.\n"
    "Do not return explanations.\n"
    "Do not wrap the JSON in code fences.\n\n"
    "The root object must be:\n\n"
    "```json\n"
    "{\n"
    '  "testCases": [],\n'
    '  "coverageNote": null\n'
    "}\n"
    "```\n\n"
    '"coverageNote" is a sibling of "testCases" on the root object, not a field on any individual test case. '
    "Each test case must use exactly these property names. "
    "These names are case-sensitive:\n\n"
    "```json\n"
    "{\n"
    '  "testCases": [\n'
    "    {\n"
    '      "requirementId": "",\n'
    '      "testCaseId": "",\n'
    '      "testCaseTitle": "",\n'
    '      "priority": "",\n'
    '      "testSuite": "",\n'
    '      "preconditions": "",\n'
    '      "steps": [\n'
    "        {\n"
    '          "stepNo": 1,\n'
    '          "action": "",\n'
    '          "expectedResult": ""\n'
    "        }\n"
    "      ],\n"
    '      "postConditions": "",\n'
    '      "testType": "",\n'
    '      "testClassification": "Smoke",\n'
    '      "workItemType": null,\n'
    '      "automationStatus": null,\n'
    '      "state": null,\n'
    '      "areaPath": null,\n'
    '      "assignedTo": null,\n'
    '      "tags": null\n'
    "    }\n"
    "  ],\n"
    '  "coverageNote": null\n'
    "}\n"
    "```\n\n"
    '"workItemType"/"automationStatus"/"state"/"areaPath"/"assignedTo"/"tags" are always `null` — see '
    '"Fields Set By The Application, Not You" above; this example shows the required JSON shape, not the '
    "values to fill in.\n\n"
    "The following are NOT valid JSON property names. Never use them — "
    "they are Azure DevOps column labels only, not JSON keys:\n\n"
    "Requirement ID, Test Case ID, Test Case Title, Priority, Test Suite, "
    "Preconditions, Step No, Action, Expected Result, Post Conditions, "
    "Automation Status, Test Type, Tags, Work Item Type, State, Area Path, Assigned To\n\n"
    "Use these JSON property names instead:\n\n"
    "requirementId, testCaseId, testCaseTitle, priority, testSuite, "
    "preconditions, steps, stepNo, action, expectedResult, postConditions, "
    "testType, testClassification, workItemType, automationStatus, state, areaPath, assignedTo, tags, "
    "coverageNote\n\n"
    "ADO column mapping (for reference only — generate the JSON property "
    "on the left, never the ADO column name on the right):\n\n"
    "| JSON Property | ADO Column |\n"
    "| --- | --- |\n"
    "| requirementId | Requirement ID |\n"
    "| testCaseId | Test Case ID |\n"
    "| testCaseTitle | Test Case Title |\n"
    "| priority | Priority |\n"
    "| testSuite | Test Suite |\n"
    "| preconditions | Preconditions |\n"
    "| stepNo | Step No |\n"
    "| action | Action |\n"
    "| expectedResult | Expected Result |\n"
    "| postConditions | Post Conditions |\n"
    "| testType | Test Type |\n"
    "| workItemType | Work Item Type |\n"
    "| automationStatus | Automation Status |\n"
    "| state | State |\n"
    "| areaPath | Area Path |\n"
    "| assignedTo | Assigned To |\n"
    "| tags | Tags |\n\n"
    "The response must validate against the application's JSON schema "
    "exactly, with no aliases and no post-processing."
)
