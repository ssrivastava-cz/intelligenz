"""Unit tests for the strongly typed AI response models — the contract
`PromptBuilder`'s Output Format section instructs the model to follow.
These validate the raw camelCase JSON shape OpenAI is expected to
return (parsed via `model_validate`), not just construction from
Python kwargs.
"""
import pytest
from pydantic import ValidationError

from app.models.generated_test_case import GeneratedTestCase, GeneratedTestCasesResponse, GeneratedTestStep

_VALID_TEST_CASE = {
    "requirementId": "REQ-1",
    "testCaseId": "TC-1",
    "testCaseTitle": "Reschedule an appointment",
    "priority": "High",
    "testSuite": "Appointments",
    "preconditions": "User is logged in.",
    "steps": [{"stepNo": 1, "action": "Open appointment.", "expectedResult": "Details are shown."}],
    "postConditions": "Appointment is updated.",
    "testType": "Functional",
    "testClassification": "Smoke",
    "workItemType": None,
    "automationStatus": None,
    "state": None,
    "areaPath": None,
    "assignedTo": None,
    "tags": None,
}


def test_generated_test_case_parses_camel_case_ai_output():
    test_case = GeneratedTestCase.model_validate(_VALID_TEST_CASE)

    assert test_case.requirement_id == "REQ-1"
    assert test_case.test_case_id == "TC-1"
    assert test_case.test_case_title == "Reschedule an appointment"
    assert test_case.priority == "High"
    assert test_case.test_suite == "Appointments"
    assert test_case.preconditions == "User is logged in."
    assert test_case.post_conditions == "Appointment is updated."
    assert test_case.test_type == "Functional"
    assert test_case.test_classification == "Smoke"


def test_generated_test_step_parses_camel_case_ai_output():
    [step] = GeneratedTestCase.model_validate(_VALID_TEST_CASE).steps

    assert isinstance(step, GeneratedTestStep)
    assert step.step_no == 1
    assert step.action == "Open appointment."
    assert step.expected_result == "Details are shown."


def test_generated_test_cases_response_parses_the_full_wrapper():
    response = GeneratedTestCasesResponse.model_validate({"testCases": [_VALID_TEST_CASE]})

    assert len(response.test_cases) == 1
    assert response.test_cases[0].test_case_id == "TC-1"


def test_generated_test_cases_response_accepts_an_empty_list():
    response = GeneratedTestCasesResponse.model_validate({"testCases": []})

    assert response.test_cases == []


@pytest.mark.parametrize("missing_field", ["requirementId", "testCaseId", "priority", "steps"])
def test_generated_test_case_rejects_a_missing_required_field(missing_field):
    payload = {key: value for key, value in _VALID_TEST_CASE.items() if key != missing_field}

    with pytest.raises(ValidationError):
        GeneratedTestCase.model_validate(payload)


def test_generated_test_case_rejects_steps_that_are_not_a_list():
    payload = {**_VALID_TEST_CASE, "steps": "Open appointment."}

    with pytest.raises(ValidationError):
        GeneratedTestCase.model_validate(payload)


def test_generated_test_case_rejects_a_step_missing_a_field():
    payload = {**_VALID_TEST_CASE, "steps": [{"stepNo": 1, "action": "Open appointment."}]}

    with pytest.raises(ValidationError):
        GeneratedTestCase.model_validate(payload)


def test_generated_test_case_rejects_tags_that_are_not_a_list_of_strings():
    payload = {**_VALID_TEST_CASE, "tags": "Appointments"}

    with pytest.raises(ValidationError):
        GeneratedTestCase.model_validate(payload)


def test_generated_test_cases_response_rejects_missing_test_cases_key():
    with pytest.raises(ValidationError):
        GeneratedTestCasesResponse.model_validate({})


def test_generated_test_cases_response_rejects_a_malformed_test_case_in_the_list():
    with pytest.raises(ValidationError):
        GeneratedTestCasesResponse.model_validate({"testCases": [{"requirementId": "REQ-1"}]})


# --- ADO fields: never required from the model, never trusted as-is ---


def test_generated_test_case_defaults_test_classification_to_regression_when_omitted():
    """A safe, conservative default for hand-authored fixtures/older
    persisted data that predate `testClassification` — the real
    generation path always gets an explicit value from
    `GenerationService._apply_ado_fields`, which never relies on this
    default in practice.
    """
    payload = {key: value for key, value in _VALID_TEST_CASE.items() if key != "testClassification"}

    test_case = GeneratedTestCase.model_validate(payload)

    assert test_case.test_classification == "Regression"


@pytest.mark.parametrize(
    "field", ["workItemType", "automationStatus", "state", "areaPath", "assignedTo", "tags"]
)
def test_generated_test_case_accepts_null_for_every_application_owned_ado_field(field):
    """The prompt instructs the model to always return `null` for these
    — Structured Outputs must accept that without complaint.
    """
    payload = {**_VALID_TEST_CASE, field: None}

    test_case = GeneratedTestCase.model_validate(payload)

    assert getattr(test_case, _CAMEL_TO_SNAKE[field]) is None


@pytest.mark.parametrize("field", ["workItemType", "automationStatus", "state", "areaPath", "assignedTo", "tags"])
def test_generated_test_case_parses_fine_when_an_application_owned_ado_field_is_entirely_missing(field):
    """Backward compatibility: a `GeneratedTestCase` persisted before
    ADO fields existed at all (the key simply absent, not `null`) still
    deserializes.
    """
    payload = {key: value for key, value in _VALID_TEST_CASE.items() if key != field}

    test_case = GeneratedTestCase.model_validate(payload)

    assert getattr(test_case, _CAMEL_TO_SNAKE[field]) is None


_CAMEL_TO_SNAKE = {
    "workItemType": "work_item_type",
    "automationStatus": "automation_status",
    "state": "state",
    "areaPath": "area_path",
    "assignedTo": "assigned_to",
    "tags": "tags",
}
