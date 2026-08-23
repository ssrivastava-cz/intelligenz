/**
 * Adapts a backend `GeneratedTestCaseOut` into the row shape the
 * existing (pre-integration) results table/Excel-export/edit UI
 * already expects — so that presentation layer stays untouched while
 * the data underneath becomes real. `postConditions`/`requirementId`/
 * `tags` have no column in that shape and are intentionally dropped
 * here: they're still present in full in the backend-generated Excel
 * download and Generation History detail, which remain the
 * authoritative record.
 */
export function mapGeneratedTestCase(testCase) {
  return {
    id: testCase.testCaseId,
    feature: testCase.testSuite,
    testCase: testCase.testCaseTitle,
    preconditions: testCase.preconditions,
    steps: testCase.steps.map((step) => step.action),
    expectedResult: testCase.steps.map((step) => step.expectedResult).join(" → "),
    priority: testCase.priority,
    type: testCase.testType,
    source: "AI Generated",
    automationCandidate: testCase.automationStatus !== "Not Automated",
  };
}

export function mapGeneratedTestCases(testCases) {
  return testCases.map(mapGeneratedTestCase);
}
