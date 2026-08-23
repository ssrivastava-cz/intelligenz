/** Column definitions shared by the results table and the Excel export. */
export const TEST_PLAN_COLUMNS = [
  { key: "id", label: "ID" },
  { key: "feature", label: "Feature" },
  { key: "testCase", label: "Test Case" },
  { key: "preconditions", label: "Preconditions" },
  { key: "steps", label: "Steps" },
  { key: "expectedResult", label: "Expected Result" },
  { key: "priority", label: "Priority" },
  { key: "type", label: "Type" },
  { key: "source", label: "Source" },
  { key: "automationCandidate", label: "Automation Candidate" },
];

export const PRIORITY_OPTIONS = ["High", "Medium", "Low"];
export const TYPE_OPTIONS = ["Functional", "Regression", "Edge Case", "Integration", "Negative"];
