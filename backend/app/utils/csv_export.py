"""Renders test cases as CSV for the /download endpoint (mock 'Excel' export)."""
import csv
import io

from app.models.test_case import TestCase

_COLUMNS = [
    ("id", "ID"),
    ("feature", "Feature"),
    ("test_case", "Test Case"),
    ("preconditions", "Preconditions"),
    ("steps", "Steps"),
    ("expected_result", "Expected Result"),
    ("priority", "Priority"),
    ("type", "Type"),
    ("source", "Source"),
    ("automation_candidate", "Automation Candidate"),
]


def test_cases_to_csv(test_cases: list[TestCase]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(label for _, label in _COLUMNS)

    for test_case in test_cases:
        row = []
        for field, _ in _COLUMNS:
            value = getattr(test_case, field)
            if isinstance(value, list):
                value = " | ".join(value)
            elif isinstance(value, bool):
                value = "Yes" if value else "No"
            elif hasattr(value, "value"):
                value = value.value
            row.append(value)
        writer.writerow(row)

    return buffer.getvalue()
