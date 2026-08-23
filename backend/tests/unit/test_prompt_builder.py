"""Unit tests for PromptBuilder — a pure formatter: given the same
structured input and timestamp, it must always produce the exact same
prompt text, with sections in a fixed order, and it must never itself
touch OpenAI, ChromaDB, Redmine, or UploadService.
"""
import inspect
from datetime import UTC, datetime

from app.models.prompt_input import PromptBuilderInput
from app.models.retrieved_chunk import RetrievedChunk
from app.services import prompt_builder as prompt_builder_module
from app.services.prompt_builder import PROMPT_VERSION, PromptBuilder

_GENERATED_AT = datetime(2026, 1, 15, 12, 30, 0, tzinfo=UTC)


def _chunk(**overrides) -> RetrievedChunk:
    base = {
        "chunk_id": "chunk-1",
        "text": "Chunk body text.",
        "similarity_score": 0.9,
        "artifact_type": "WORKFLOW",
        "feature": "Appointments",
        "source_filename": "onboarding.md",
        "section_heading": "Role Permissions",
        "page_number": None,
        "collection_name": "source_of_truth_chunks",
    }
    base.update(overrides)
    return RetrievedChunk(**base)


def _input(**overrides) -> PromptBuilderInput:
    base = dict(
        feature="Appointments",
        redmine_ticket="12345",
        redmine_description="Users should be able to reschedule appointments.",
        user_description=None,
        workflow_chunks=[],
        historical_test_case_chunks=[],
        historical_issue_chunks=[],
        uploaded_document_chunks=[],
    )
    base.update(overrides)
    return PromptBuilderInput(**base)


def test_prompt_builder_module_never_imports_openai_chromadb_redmine_or_upload_service():
    source = inspect.getsource(prompt_builder_module)
    forbidden_names = (
        "openai",
        "chromadb",
        "RedmineService",
        "UploadService",
        "VectorStoreService",
        "EmbeddingService",
    )
    for forbidden in forbidden_names:
        assert forbidden not in source


def test_build_is_deterministic_for_the_same_input_and_timestamp():
    builder = PromptBuilder()
    prompt_input = _input(
        workflow_chunks=[_chunk(chunk_id="a")],
        historical_test_case_chunks=[_chunk(chunk_id="b", artifact_type="TEST_CASE")],
    )

    first = builder.build(prompt_input, generated_at=_GENERATED_AT)
    second = builder.build(prompt_input, generated_at=_GENERATED_AT)

    assert first.prompt_text == second.prompt_text
    assert first == second


def test_build_does_not_mutate_the_input():
    builder = PromptBuilder()
    workflow_chunks = [_chunk(chunk_id="a")]
    prompt_input = _input(workflow_chunks=workflow_chunks)

    builder.build(prompt_input, generated_at=_GENERATED_AT)

    assert prompt_input.workflow_chunks == workflow_chunks


def test_header_includes_prompt_version_and_generated_timestamp():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)

    assert result.prompt_version == PROMPT_VERSION
    assert result.generated_at == _GENERATED_AT
    assert f"Prompt Version: {PROMPT_VERSION}" in result.sections.header
    assert _GENERATED_AT.isoformat() in result.sections.header
    assert result.prompt_text.startswith(result.sections.header)


def test_sections_appear_in_the_specified_order():
    builder = PromptBuilder()
    prompt_input = _input(
        workflow_chunks=[_chunk(chunk_id="w")],
        historical_test_case_chunks=[_chunk(chunk_id="t", artifact_type="TEST_CASE")],
        historical_issue_chunks=[_chunk(chunk_id="i", artifact_type="ISSUE")],
        uploaded_document_chunks=[_chunk(chunk_id="u", artifact_type="USER_UPLOAD")],
    )

    result = builder.build(prompt_input, generated_at=_GENERATED_AT)
    text = result.prompt_text

    markers = [
        f"Prompt Version: {PROMPT_VERSION}",
        "## System Instructions",
        "## Feature",
        "## Redmine Description",
        "## Uploaded Documents",
        "## Workflow Context",
        "## Historical Test Cases",
        "## Historical Issues",
        "## Test Case Generation Instructions",
        "## Generation Guidelines",
        "## Output Format",
    ]
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions)


def test_system_instructions_define_role_and_priority_order():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    instructions = result.sections.system_instructions

    assert "Senior QA Automation Engineer" in instructions
    assert "Azure DevOps (ADO)" in instructions
    # Several phrases (e.g. "Workflow Documents") appear twice — once in
    # the responsibilities list, once in the priority list — so ordering
    # is only checked within the priority list itself.
    priority_list = instructions[instructions.index("prioritize in this order") :]
    priority_positions = [
        priority_list.index("Uploaded Documents"),
        priority_list.index("Redmine Description"),
        priority_list.index("Workflow Documents"),
        priority_list.index("Historical Test Cases"),
        priority_list.index("Historical Issues"),
    ]
    assert priority_positions == sorted(priority_positions)
    assert "Never invent functionality" in instructions


def test_feature_section_includes_feature_name_and_redmine_ticket():
    builder = PromptBuilder()

    result = builder.build(_input(feature="Appointments", redmine_ticket="12345"), generated_at=_GENERATED_AT)

    assert "Feature Name: Appointments" in result.sections.feature_section
    assert "Redmine Ticket: 12345" in result.sections.feature_section


def test_feature_section_omits_redmine_ticket_line_when_not_supplied():
    """Generation can run from the Feature alone — no Redmine ticket
    reference should appear when none was given.
    """
    builder = PromptBuilder()

    result = builder.build(_input(redmine_ticket=""), generated_at=_GENERATED_AT)

    assert "Redmine Ticket" not in result.sections.feature_section


def test_feature_section_omits_user_description_when_not_supplied():
    builder = PromptBuilder()

    result = builder.build(_input(user_description=None), generated_at=_GENERATED_AT)

    assert "User Description" not in result.sections.feature_section


def test_feature_section_includes_user_description_when_supplied():
    builder = PromptBuilder()

    result = builder.build(
        _input(user_description="Focus on the reschedule flow only."), generated_at=_GENERATED_AT
    )

    assert "User Description: Focus on the reschedule flow only." in result.sections.feature_section


def test_redmine_description_is_inserted_exactly_as_provided():
    builder = PromptBuilder()
    description = "Line one.\nLine two with *odd* formatting -- and trailing spaces.   "

    result = builder.build(_input(redmine_description=description), generated_at=_GENERATED_AT)

    assert description in result.sections.redmine_description_section


def test_redmine_description_section_shows_a_placeholder_when_not_supplied():
    builder = PromptBuilder()

    result = builder.build(_input(redmine_description=""), generated_at=_GENERATED_AT)

    assert "No Redmine ticket description was provided" in result.sections.redmine_description_section


def test_empty_chunk_sections_render_a_placeholder_message():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)

    assert "No uploaded documents were retrieved" in result.sections.uploaded_documents_section
    assert "No workflow sections were retrieved" in result.sections.workflow_section
    assert "No historical test cases were retrieved" in result.sections.historical_test_cases_section
    assert "No historical issues were retrieved" in result.sections.historical_issues_section
    # Still no chunk headers show up despite the empty state.
    assert "###" not in result.sections.workflow_section


def test_historical_issues_section_explains_regression_purpose_even_when_empty():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)

    assert "previous defects" in result.sections.historical_issues_section
    assert "regression coverage" in result.sections.historical_issues_section


def test_workflow_section_preserves_order_headings_and_document_names():
    builder = PromptBuilder()
    chunks = [
        _chunk(chunk_id="first", source_filename="a.md", section_heading="Section A", text="Body A."),
        _chunk(chunk_id="second", source_filename="b.md", section_heading="Section B", text="Body B."),
        _chunk(chunk_id="third", source_filename="c.md", section_heading="Section C", text="Body C."),
    ]

    result = builder.build(_input(workflow_chunks=chunks), generated_at=_GENERATED_AT)
    section = result.sections.workflow_section

    positions = [section.index("Body A."), section.index("Body B."), section.index("Body C.")]
    assert positions == sorted(positions)
    for chunk in chunks:
        assert chunk.source_filename in section
        assert chunk.section_heading in section
        assert chunk.text in section


def test_chunk_section_omits_section_heading_line_when_none():
    builder = PromptBuilder()
    chunk = _chunk(section_heading=None, text="Body without a heading.")

    result = builder.build(_input(workflow_chunks=[chunk]), generated_at=_GENERATED_AT)

    assert "Section: " not in result.sections.workflow_section
    assert "Body without a heading." in result.sections.workflow_section


def test_mixed_retrieval_keeps_each_category_in_its_own_section():
    builder = PromptBuilder()
    prompt_input = _input(
        workflow_chunks=[_chunk(chunk_id="w", text="Workflow body.", artifact_type="WORKFLOW")],
        historical_test_case_chunks=[_chunk(chunk_id="t", text="Test case body.", artifact_type="TEST_CASE")],
        historical_issue_chunks=[_chunk(chunk_id="i", text="Issue body.", artifact_type="ISSUE")],
        uploaded_document_chunks=[_chunk(chunk_id="u", text="Upload body.", artifact_type="USER_UPLOAD")],
    )

    result = builder.build(prompt_input, generated_at=_GENERATED_AT)

    assert "Workflow body." in result.sections.workflow_section
    assert "Workflow body." not in result.sections.historical_test_cases_section
    assert "Test case body." in result.sections.historical_test_cases_section
    assert "Test case body." not in result.sections.workflow_section
    assert "Issue body." in result.sections.historical_issues_section
    assert "Upload body." in result.sections.uploaded_documents_section


def test_test_case_generation_instructions_state_the_default_max_test_cases_as_an_exact_target():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)

    assert "Generate exactly 5 test cases." in result.sections.test_case_generation_instructions


def test_test_case_generation_instructions_reflect_a_configured_max_test_cases():
    """The limit is read from configuration (via the constructor), never
    hardcoded — a different configured value must show up verbatim, as
    an exact target rather than a mere ceiling.
    """
    builder = PromptBuilder(max_test_cases=10)

    result = builder.build(_input(), generated_at=_GENERATED_AT)

    assert "Generate exactly 10 test cases." in result.sections.test_case_generation_instructions
    assert "Generate exactly 5 test cases." not in result.sections.test_case_generation_instructions


def test_test_case_generation_instructions_never_exceed_the_configured_maximum():
    builder = PromptBuilder(max_test_cases=10)

    result = builder.build(_input(), generated_at=_GENERATED_AT)

    assert "Never generate more than 10 test cases." in result.sections.test_case_generation_instructions


def test_test_case_generation_instructions_forbid_reaching_the_count_via_duplication_or_fabrication():
    builder = PromptBuilder(max_test_cases=10)

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    instructions = result.sections.test_case_generation_instructions

    assert "duplicating a test case" in instructions
    assert "fabricating a scenario the retrieved context does not support" in instructions


def test_test_case_generation_instructions_allow_fewer_test_cases_when_evidence_is_limited():
    builder = PromptBuilder(max_test_cases=10)

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    instructions = result.sections.test_case_generation_instructions

    assert "generate that smaller number instead" in instructions
    assert "coverageNote" in instructions


def test_output_format_documents_the_coverage_note_field():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)

    assert '"coverageNote"' in result.sections.output_format


def test_test_case_generation_instructions_lists_every_required_column():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    instructions = result.sections.test_case_generation_instructions

    for column in (
        "Requirement ID",
        "Test Case ID",
        "Test Case Title",
        "Priority",
        "Test Suite",
        "Preconditions",
        "Step No.",
        "Action",
        "Expected Result",
        "Post Conditions",
        "Automation Status",
        "Test Type",
        "Tags",
    ):
        assert column in instructions


def test_output_format_requires_json_and_forbids_other_formats():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    output_format = result.sections.output_format

    assert "Return only valid JSON." in output_format
    assert "Do not return Markdown." in output_format
    assert "Do not return tables." in output_format
    assert "Do not return explanations." in output_format
    assert "Do not wrap the JSON in code fences." in output_format


def test_output_format_specifies_the_exact_camel_case_json_property_names():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    output_format = result.sections.output_format

    for json_property in (
        '"requirementId"',
        '"testCaseId"',
        '"testCaseTitle"',
        '"priority"',
        '"testSuite"',
        '"preconditions"',
        '"steps"',
        '"stepNo"',
        '"action"',
        '"expectedResult"',
        '"postConditions"',
        '"automationStatus"',
        '"testType"',
        '"tags"',
    ):
        assert json_property in output_format


def test_output_format_explicitly_forbids_ado_display_labels_as_json_keys():
    """Regression test for a real failure mode: the model returning ADO
    display column names ("Requirement ID") instead of the required
    camelCase JSON property names ("requirementId"), which fails
    Pydantic validation. The prompt must explicitly rule this out.
    """
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    output_format = result.sections.output_format

    assert "NOT valid JSON property names" in output_format
    for ado_label in (
        "Requirement ID",
        "Test Case ID",
        "Test Case Title",
        "Test Suite",
        "Step No",
        "Expected Result",
        "Post Conditions",
        "Automation Status",
        "Test Type",
    ):
        assert ado_label in output_format


# --- ADO structure: one scenario = one test case, multiple ordered steps ---


def test_instructions_forbid_a_separate_test_case_per_step():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    instructions = result.sections.test_case_generation_instructions

    assert "one test scenario" in instructions
    assert "Do NOT create a separate test case for every step" in instructions


def test_instructions_require_step_numbering_to_restart_per_test_case():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    instructions = result.sections.test_case_generation_instructions

    assert "restart" in instructions.lower()


def test_instructions_forbid_vague_step_actions():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    instructions = result.sections.test_case_generation_instructions

    assert "Verify functionality" in instructions
    assert "Click the Delete option from the kebab menu" in instructions


def test_instructions_require_independent_meaningful_test_cases():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)

    assert "independent and meaningful" in result.sections.test_case_generation_instructions


# --- Test Case Classification: Smoke vs. Regression ---


def test_instructions_require_exactly_smoke_or_regression_classification():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    instructions = result.sections.test_case_generation_instructions

    assert '"testClassification"' in instructions
    assert '"Smoke"' in instructions
    assert '"Regression"' in instructions


def test_instructions_describe_smoke_and_regression_criteria():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    instructions = result.sections.test_case_generation_instructions

    assert "happy-path" in instructions
    assert "edge cases" in instructions.lower()
    assert "negative scenarios" in instructions.lower()


# --- Fields the application sets, never the model ---


def test_instructions_tell_the_model_to_null_every_application_owned_ado_field():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    instructions = result.sections.test_case_generation_instructions

    for field in ('"workItemType"', '"automationStatus"', '"state"', '"areaPath"', '"assignedTo"', '"tags"'):
        assert field in instructions
    assert "assigns every one of these after generation" in instructions
    assert "never from anything you invent" in instructions


def test_instructions_state_automation_status_is_always_not_automated():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)

    assert '"Not Automated"' in result.sections.test_case_generation_instructions


def test_output_format_json_example_includes_every_ado_field():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    output_format = result.sections.output_format

    for key in (
        '"testClassification"',
        '"workItemType"',
        '"automationStatus"',
        '"state"',
        '"areaPath"',
        '"assignedTo"',
    ):
        assert key in output_format


def test_output_format_forbids_the_new_ado_column_labels_as_json_keys():
    builder = PromptBuilder()

    result = builder.build(_input(), generated_at=_GENERATED_AT)
    output_format = result.sections.output_format

    for ado_label in ("Work Item Type", "State", "Area Path", "Assigned To"):
        assert ado_label in output_format
