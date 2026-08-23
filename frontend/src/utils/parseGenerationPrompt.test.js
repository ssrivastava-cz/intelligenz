import { describe, expect, it } from "vitest";

import { parseGenerationQuery } from "./parseGenerationPrompt.js";

function buildPrompt({ userDescriptionLine = "", redmineDescription = "Some redmine description." } = {}) {
  return [
    "Prompt Version: 1.0.0",
    "Generated At: 2026-08-03T12:00:00+00:00",
    "",
    "## System Instructions",
    "",
    "You are a Senior QA Automation Engineer.",
    "",
    "## Feature",
    "",
    "Feature Name: Contact Log",
    "Redmine Ticket: 54980",
    ...(userDescriptionLine ? [userDescriptionLine] : []),
    "",
    "## Redmine Description",
    "",
    redmineDescription,
    "",
    "## Uploaded Documents",
    "",
    "_No uploaded documents were retrieved for this feature._",
  ].join("\n");
}

describe("parseGenerationQuery", () => {
  it("extracts the Redmine description section", () => {
    const prompt = buildPrompt({ redmineDescription: "Optum requested a Bridge indicator." });

    const { redmineDescription } = parseGenerationQuery(prompt);

    expect(redmineDescription).toBe("Optum requested a Bridge indicator.");
  });

  it("extracts a multi-line Redmine description in full", () => {
    const prompt = buildPrompt({
      redmineDescription: "Optum requested a Bridge indicator to identify\nbridged Contact Log records.",
    });

    const { redmineDescription } = parseGenerationQuery(prompt);

    expect(redmineDescription).toBe("Optum requested a Bridge indicator to identify\nbridged Contact Log records.");
  });

  it("extracts a single-line user description", () => {
    const prompt = buildPrompt({ userDescriptionLine: "User Description: Focus on UI filtering." });

    const { userDescription } = parseGenerationQuery(prompt);

    expect(userDescription).toBe("Focus on UI filtering.");
  });

  it("extracts a multi-line user description in full", () => {
    // The "User Description:" line and its continuation both land in
    // the array before the join, since PromptBuilder writes the raw
    // (possibly multi-line) string directly after the label.
    const prompt = buildPrompt({
      userDescriptionLine: "User Description: Generate regression test cases with emphasis\non UI filtering.",
    });

    const { userDescription } = parseGenerationQuery(prompt);

    expect(userDescription).toBe("Generate regression test cases with emphasis\non UI filtering.");
  });

  it("returns null user description when none was given", () => {
    const prompt = buildPrompt({ userDescriptionLine: "" });

    const { userDescription } = parseGenerationQuery(prompt);

    expect(userDescription).toBeNull();
  });

  it("returns null for both fields when the prompt is empty or missing", () => {
    expect(parseGenerationQuery("")).toEqual({ redmineDescription: null, userDescription: null });
    expect(parseGenerationQuery(null)).toEqual({ redmineDescription: null, userDescription: null });
    expect(parseGenerationQuery(undefined)).toEqual({ redmineDescription: null, userDescription: null });
  });

  it("returns null redmine description when the heading is missing entirely", () => {
    const prompt = "Prompt Version: 1.0.0\n\n## Feature\n\nFeature Name: X\nRedmine Ticket: 1";

    const { redmineDescription } = parseGenerationQuery(prompt);

    expect(redmineDescription).toBeNull();
  });
});
