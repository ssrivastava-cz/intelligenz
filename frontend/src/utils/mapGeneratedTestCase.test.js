import { describe, expect, it } from "vitest";

import { mapGeneratedTestCase, mapGeneratedTestCases } from "./mapGeneratedTestCase.js";

function sampleTestCase(overrides = {}) {
  return {
    requirementId: "REQ-1",
    testCaseId: "TC-1",
    testCaseTitle: "Reschedule an appointment",
    priority: "High",
    testSuite: "Appointments",
    preconditions: "User is logged in.",
    steps: [
      { stepNo: 1, action: "Open appointment.", expectedResult: "Details are shown." },
      { stepNo: 2, action: "Click reschedule.", expectedResult: "Reschedule form appears." },
    ],
    postConditions: "Appointment is updated.",
    automationStatus: "Not Automated",
    testType: "Functional",
    tags: ["Appointments"],
    ...overrides,
  };
}

describe("mapGeneratedTestCase", () => {
  it("maps identifying and categorical fields", () => {
    const row = mapGeneratedTestCase(sampleTestCase());

    expect(row.id).toBe("TC-1");
    expect(row.feature).toBe("Appointments");
    expect(row.testCase).toBe("Reschedule an appointment");
    expect(row.preconditions).toBe("User is logged in.");
    expect(row.priority).toBe("High");
    expect(row.type).toBe("Functional");
    expect(row.source).toBe("AI Generated");
  });

  it("flattens step actions into a plain string array", () => {
    const row = mapGeneratedTestCase(sampleTestCase());

    expect(row.steps).toEqual(["Open appointment.", "Click reschedule."]);
  });

  it("joins every step's expected result into one string", () => {
    const row = mapGeneratedTestCase(sampleTestCase());

    expect(row.expectedResult).toBe("Details are shown. → Reschedule form appears.");
  });

  it("marks automationCandidate false when automationStatus is Not Automated", () => {
    const row = mapGeneratedTestCase(sampleTestCase({ automationStatus: "Not Automated" }));

    expect(row.automationCandidate).toBe(false);
  });

  it("marks automationCandidate true for any other automationStatus value", () => {
    const row = mapGeneratedTestCase(sampleTestCase({ automationStatus: "Automated" }));

    expect(row.automationCandidate).toBe(true);
  });

  it("mapGeneratedTestCases maps every item in the list", () => {
    const rows = mapGeneratedTestCases([sampleTestCase({ testCaseId: "TC-1" }), sampleTestCase({ testCaseId: "TC-2" })]);

    expect(rows.map((row) => row.id)).toEqual(["TC-1", "TC-2"]);
  });
});
