"""Static mock fixtures shared across services.

No AI/DB integration — this is the single source of realistic-looking
sample data the mock endpoints draw from.
"""
from datetime import UTC, datetime, timedelta

from app.models.common import GenerationStatus, Priority, TestCaseSource, TestType
from app.models.test_case import TestCase

PROGRESS_STAGES = [
    "Reading Redmine",
    "Reading Documents",
    "Generating Embeddings",
    "Searching Knowledge Base",
    "Building Prompt",
    "Generating Test Plan",
    "Finalizing",
]

_SAMPLE_TEST_CASES = [
    {
        "id": "TC-001",
        "test_case": "Book a new appointment with valid details",
        "preconditions": "User is logged in and has at least one active provider assigned.",
        "steps": [
            "Navigate to the Appointments module",
            "Click 'New Appointment'",
            "Select a provider, date, and time slot",
            "Fill in patient details and reason for visit",
            "Click 'Save'",
        ],
        "expected_result": "Appointment is created and appears in the provider's calendar with status 'Scheduled'.",
        "priority": Priority.HIGH,
        "type": TestType.FUNCTIONAL,
        "source": TestCaseSource.REDMINE_TICKET,
        "automation_candidate": True,
    },
    {
        "id": "TC-002",
        "test_case": "Prevent double-booking the same time slot",
        "preconditions": "A time slot is already booked for the selected provider.",
        "steps": [
            "Attempt to book a new appointment for the same provider, date, and time",
            "Click 'Save'",
        ],
        "expected_result": "System blocks the booking and displays a conflict warning.",
        "priority": Priority.HIGH,
        "type": TestType.EDGE_CASE,
        "source": TestCaseSource.HISTORICAL_TEST_CASE,
        "automation_candidate": True,
    },
    {
        "id": "TC-003",
        "test_case": "Reschedule an existing appointment",
        "preconditions": "An upcoming appointment exists for the patient.",
        "steps": [
            "Open the existing appointment",
            "Click 'Reschedule'",
            "Select a new date and time",
            "Confirm the change",
        ],
        "expected_result": "Appointment is updated to the new date/time and the patient receives a notification.",
        "priority": Priority.MEDIUM,
        "type": TestType.FUNCTIONAL,
        "source": TestCaseSource.HISTORICAL_TEST_CASE,
        "automation_candidate": True,
    },
    {
        "id": "TC-004",
        "test_case": "Cancel an appointment within the allowed window",
        "preconditions": "Appointment is more than 24 hours in the future.",
        "steps": ["Open the appointment", "Click 'Cancel'", "Confirm cancellation"],
        "expected_result": "Appointment status changes to 'Cancelled' and the slot is released.",
        "priority": Priority.MEDIUM,
        "type": TestType.FUNCTIONAL,
        "source": TestCaseSource.RELEASE_NOTES,
        "automation_candidate": False,
    },
    {
        "id": "TC-005",
        "test_case": "Reject cancellation outside the allowed window",
        "preconditions": "Appointment starts in under 24 hours.",
        "steps": ["Open the appointment", "Click 'Cancel'"],
        "expected_result": "System displays a policy message and prevents cancellation.",
        "priority": Priority.LOW,
        "type": TestType.NEGATIVE,
        "source": TestCaseSource.HISTORICAL_ISSUE_SHEET,
        "automation_candidate": False,
    },
    {
        "id": "TC-006",
        "test_case": "Send reminder notification 24 hours before appointment",
        "preconditions": "Appointment is scheduled for tomorrow at the same time.",
        "steps": ["Wait for the scheduled reminder job to run", "Check patient's notification inbox"],
        "expected_result": "A reminder notification is sent exactly 24 hours before the appointment.",
        "priority": Priority.MEDIUM,
        "type": TestType.INTEGRATION,
        "source": TestCaseSource.GENERATED,
        "automation_candidate": True,
    },
    {
        "id": "TC-007",
        "test_case": "Filter appointments list by provider and date range",
        "preconditions": "Multiple appointments exist across providers and dates.",
        "steps": [
            "Open the Appointments list view",
            "Apply a provider filter",
            "Apply a date range filter",
            "Review the filtered results",
        ],
        "expected_result": "Only appointments matching both filters are displayed.",
        "priority": Priority.LOW,
        "type": TestType.FUNCTIONAL,
        "source": TestCaseSource.HISTORICAL_TEST_CASE,
        "automation_candidate": True,
    },
    {
        "id": "TC-008",
        "test_case": "Validate required fields on appointment form",
        "preconditions": "New appointment form is open.",
        "steps": ["Leave required fields (provider, date, time) empty", "Click 'Save'"],
        "expected_result": "Form displays inline validation errors and does not submit.",
        "priority": Priority.MEDIUM,
        "type": TestType.NEGATIVE,
        "source": TestCaseSource.GENERATED,
        "automation_candidate": True,
    },
]


def build_mock_test_cases(feature: str) -> list[TestCase]:
    """Returns a fresh set of sample test cases labeled for the given feature."""
    return [TestCase(feature=feature, **fields) for fields in _SAMPLE_TEST_CASES]


def _seed_generation(
    generation_id: str,
    feature: str,
    redmine_id: str,
    days_ago: float,
    status: GenerationStatus,
) -> dict:
    created_at = datetime.now(UTC) - timedelta(days=days_ago)
    test_cases = build_mock_test_cases(feature) if status == GenerationStatus.COMPLETED else []
    return {
        "id": generation_id,
        "feature": feature,
        "redmine_id": redmine_id,
        "description": None,
        "session_id": None,
        "document_ids": [],
        "analysis_options": [],
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "stage_index": len(PROGRESS_STAGES) - 1,
        "test_cases": test_cases,
    }


SEED_GENERATIONS = [
    _seed_generation("gen-seed-1042", "Appointments", "12345", 0.2, GenerationStatus.COMPLETED),
    _seed_generation("gen-seed-1041", "Coding Tool", "12310,12311", 0.6, GenerationStatus.COMPLETED),
    _seed_generation("gen-seed-1040", "Analytics", "12298", 1.3, GenerationStatus.COMPLETED),
    _seed_generation("gen-seed-1039", "Contact Log", "12276", 1.7, GenerationStatus.FAILED),
    _seed_generation("gen-seed-1038", "Sticket Log", "12250", 2.4, GenerationStatus.COMPLETED),
]
