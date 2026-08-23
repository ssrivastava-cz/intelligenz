import { describe, expect, it } from "vitest";

import { buildStageKeys, buildStageLabels, resolveCurrentStageIndex } from "./generationStages.js";

describe("generationStages", () => {
  it("includes the Redmine ticket stage first when a ticket id was supplied", () => {
    const keys = buildStageKeys(true);

    expect(keys[0]).toBe("fetchingTicket");
    expect(keys).toContain("retrievingDocuments");
    expect(keys).toContain("complete");
  });

  it("omits the Redmine ticket stage entirely when no ticket id was supplied", () => {
    const keys = buildStageKeys(false);

    expect(keys).not.toContain("fetchingTicket");
    expect(keys[0]).toBe("retrievingDocuments");
  });

  it("buildStageLabels returns one human-readable label per stage key, in the same order", () => {
    const keys = buildStageKeys(true);
    const labels = buildStageLabels(true);

    expect(labels).toHaveLength(keys.length);
    expect(labels[0]).toBe("Fetching Redmine ticket details");
    expect(labels[labels.length - 1]).toBe("Generation complete");
  });

  it("resolveCurrentStageIndex finds the index of the current stage key", () => {
    expect(resolveCurrentStageIndex(true, "fetchingTicket")).toBe(0);
    expect(resolveCurrentStageIndex(true, "generatingTestCases")).toBe(3);
  });

  it("resolveCurrentStageIndex accounts for the ticket stage being absent", () => {
    // With no ticket stage, "retrievingDocuments" moves up to index 0.
    expect(resolveCurrentStageIndex(false, "retrievingDocuments")).toBe(0);
  });

  it("resolveCurrentStageIndex falls back to 0 for an unknown or missing key", () => {
    expect(resolveCurrentStageIndex(true, null)).toBe(0);
    expect(resolveCurrentStageIndex(true, "somethingUnexpected")).toBe(0);
  });
});
