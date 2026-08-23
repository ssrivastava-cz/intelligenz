import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import GenerationQuerySummary from "./GenerationQuerySummary.jsx";

function buildPrompt({ userDescriptionLine = "", redmineDescription = "Some redmine description." } = {}) {
  return [
    "Prompt Version: 1.0.0",
    "Generated At: 2026-08-03T12:00:00+00:00",
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

function buildDetail(overrides = {}) {
  return {
    feature: "Contact Log",
    redmineTicket: "54980",
    prompt: buildPrompt(),
    metadata: { topK: 5, uploadSessionId: "sess-fa91594568", generatedTestCases: 18 },
    ...overrides,
  };
}

describe("GenerationQuerySummary", () => {
  it("renders feature, redmine ticket, upload session, and both descriptions", () => {
    const detail = buildDetail({
      prompt: buildPrompt({
        redmineDescription: "Optum requested a Bridge indicator to identify bridged Contact Log records.",
        userDescriptionLine: "User Description: Generate regression test cases with emphasis on UI filtering.",
      }),
    });

    render(<GenerationQuerySummary detail={detail} />);

    expect(screen.getByText("Generation Query")).toBeInTheDocument();
    expect(screen.getByText("Contact Log")).toBeInTheDocument();
    expect(screen.getByText("54980")).toBeInTheDocument();
    expect(screen.getByText("sess-fa91594568")).toBeInTheDocument();
    expect(screen.getByText(/Optum requested a Bridge indicator/)).toBeInTheDocument();
    expect(screen.getByText(/Generate regression test cases/)).toBeInTheDocument();
  });

  it("shows a fallback message when user description is empty", () => {
    const detail = buildDetail({ prompt: buildPrompt({ userDescriptionLine: "" }) });

    render(<GenerationQuerySummary detail={detail} />);

    expect(screen.getByText("No additional description provided.")).toBeInTheDocument();
  });

  it("omits the Upload Session field when it is missing", () => {
    const detail = buildDetail({ metadata: { topK: 5, uploadSessionId: null, generatedTestCases: 18 } });

    render(<GenerationQuerySummary detail={detail} />);

    expect(screen.queryByText("Upload Session")).not.toBeInTheDocument();
  });

  it("still renders feature and ticket when other optional fields are missing", () => {
    const detail = buildDetail({
      metadata: { topK: null, uploadSessionId: null, generatedTestCases: 0 },
      prompt: buildPrompt({ userDescriptionLine: "" }),
    });

    render(<GenerationQuerySummary detail={detail} />);

    expect(screen.getByText("Contact Log")).toBeInTheDocument();
    expect(screen.getByText("54980")).toBeInTheDocument();
    expect(screen.queryByText("Upload Session")).not.toBeInTheDocument();
    expect(screen.getByText("No additional description provided.")).toBeInTheDocument();
  });

  it("allows expanding and collapsing a long Redmine description", async () => {
    const user = userEvent.setup();
    const longDescription = Array.from({ length: 15 }, (_, i) => `Line ${i + 1} of the ticket description.`).join(
      "\n",
    );
    const detail = buildDetail({ prompt: buildPrompt({ redmineDescription: longDescription }) });

    render(<GenerationQuerySummary detail={detail} />);

    const [redmineToggle] = screen.getAllByRole("button", { name: "Show More" });
    await user.click(redmineToggle);

    expect(screen.getAllByRole("button", { name: "Show Less" }).length).toBeGreaterThan(0);
    expect(screen.getByText(/Line 15 of the ticket description\./)).toBeInTheDocument();
  });

  it("shows a fallback message when no Redmine description can be recovered", () => {
    const detail = buildDetail({ prompt: "Prompt Version: 1.0.0\n\n## Feature\n\nFeature Name: X" });

    render(<GenerationQuerySummary detail={detail} />);

    expect(screen.getByText("No Redmine description available.")).toBeInTheDocument();
  });
});
