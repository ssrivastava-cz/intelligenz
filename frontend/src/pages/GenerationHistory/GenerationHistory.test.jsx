import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { generationHistoryApi } from "../../api/generationHistoryApi.js";
import GenerationHistory from "./GenerationHistory.jsx";

vi.mock("../../api/generationHistoryApi.js", () => ({
  generationHistoryApi: {
    list: vi.fn(),
    stats: vi.fn(),
    getById: vi.fn(),
    downloadExcel: vi.fn(),
  },
}));

const SAMPLE_SUMMARY = {
  generationId: "2026-08-03T10-00-00",
  createdAt: "2026-08-03T10:00:00Z",
  feature: "Contact Log",
  redmineTicket: "54980",
  model: "gpt-5",
  promptVersion: "1.0.0",
  generationTimeMs: 7200,
  numberOfTestCases: 18,
  promptTokens: 22128,
  completionTokens: 7615,
  totalTokens: 29743,
  totalAiCostUsd: 0.1038,
  totalAiCostInr: 2.43,
  status: "SUCCESS",
};

const SAMPLE_STATS = {
  totalGenerations: 2,
  successfulGenerations: 2,
  failedGenerations: 0,
  averageGenerationTimeMs: 6500,
  averageTestCases: 14.5,
  averagePromptTokens: 20000,
  averageCompletionTokens: 6000,
  averageCostUsd: 0.09,
  averageCostInr: 2.17,
  mostUsedFeature: "Contact Log",
  mostUsedModel: "gpt-5",
  totalSpendUsd: 0.18,
  totalSpendInr: 4.35,
};

const SAMPLE_DETAIL = {
  generationId: "2026-08-03T10-00-00",
  timestamp: "2026-08-03T10:00:00Z",
  feature: "Contact Log",
  redmineTicket: "54980",
  model: "gpt-5",
  promptVersion: "1.0.0",
  retrieval: { workflowChunks: 28, historicalTestCases: 14, historicalIssues: 100, uploadedDocuments: 1 },
  estimatedUsage: { model: "gpt-5", inputTokens: 22111, estimatedInputCost: { usd: 0.0276, inr: 2.63 } },
  actualUsage: {
    model: "gpt-5",
    promptTokens: 22128,
    completionTokens: 7615,
    totalTokens: 29743,
    inputCost: { usd: 0.0277, inr: 2.63 },
    outputCost: { usd: 0.0762, inr: 7.23 },
    totalCost: { usd: 0.1038, inr: 9.86 },
  },
  pricing: { model: "gpt-5", inputPricePerMillionTokens: 1.25, outputPricePerMillionTokens: 10, usdToInrExchangeRate: 95 },
  generationTimeMs: 7200,
  prompt: [
    "Prompt Version: 1.0.0",
    "Generated At: 2026-08-03T10:00:00+00:00",
    "",
    "## Feature",
    "",
    "Feature Name: Contact Log",
    "Redmine Ticket: 54980",
    "User Description: Generate regression test cases with emphasis on UI filtering.",
    "",
    "## Redmine Description",
    "",
    "Optum requested a Bridge indicator to identify bridged Contact Log records.",
    "",
    "## Uploaded Documents",
    "",
    "_No uploaded documents were retrieved for this feature._",
  ].join("\n"),
  response: "irrelevant to this UI",
  testCases: [
    { testCaseId: "TC-001", testCaseTitle: "Log a new contact" },
    { testCaseId: "TC-002", testCaseTitle: "Search existing contacts" },
  ],
  metadata: { topK: 5, uploadSessionId: null, generatedTestCases: 2 },
};

function renderPage() {
  return render(<GenerationHistory />);
}

describe("GenerationHistory page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows a loading state while history and stats are being fetched", () => {
    generationHistoryApi.list.mockReturnValue(new Promise(() => {}));
    generationHistoryApi.stats.mockReturnValue(new Promise(() => {}));

    renderPage();

    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("loads and displays generation history rows", async () => {
    generationHistoryApi.list.mockResolvedValue({ items: [SAMPLE_SUMMARY], page: 1, pageSize: 50, totalItems: 1 });
    generationHistoryApi.stats.mockResolvedValue(SAMPLE_STATS);

    renderPage();

    expect(await screen.findByText("Contact Log")).toBeInTheDocument();
    expect(screen.getByText("54980")).toBeInTheDocument();
    expect(screen.getByText("18")).toBeInTheDocument();
    expect(screen.getByText("₹2.43")).toBeInTheDocument();
    expect(screen.getByText("7.2 sec")).toBeInTheDocument();
  });

  it('shows "Test Plan Generation History" as the section heading', async () => {
    generationHistoryApi.list.mockResolvedValue({ items: [SAMPLE_SUMMARY], page: 1, pageSize: 50, totalItems: 1 });
    generationHistoryApi.stats.mockResolvedValue(SAMPLE_STATS);

    renderPage();

    expect(await screen.findByText("Test Plan Generation History")).toBeInTheDocument();
  });

  it("displays each row's actual per-generation INR cost, not a $0.00 fallback", async () => {
    // Regression test: the table must read the same field the backend
    // already sums for the Total Spend card (`totalAiCostInr`) — not a
    // `totalCostInr` field the API never returns, which silently
    // rendered as ₹0.00 for every row.
    const generationWithDistinctCost = { ...SAMPLE_SUMMARY, generationId: "gen-distinct-cost", totalAiCostInr: 7.48 };
    generationHistoryApi.list.mockResolvedValue({
      items: [generationWithDistinctCost],
      page: 1,
      pageSize: 50,
      totalItems: 1,
    });
    generationHistoryApi.stats.mockResolvedValue(SAMPLE_STATS);

    renderPage();

    expect(await screen.findByText("₹7.48")).toBeInTheDocument();
    expect(screen.queryByText("₹0.00")).not.toBeInTheDocument();
  });

  it("loads and displays dashboard statistics", async () => {
    generationHistoryApi.list.mockResolvedValue({ items: [SAMPLE_SUMMARY] });
    generationHistoryApi.stats.mockResolvedValue(SAMPLE_STATS);

    renderPage();

    await screen.findByText("Contact Log");

    expect(screen.getByText("Total Generations")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("Total Spend")).toBeInTheDocument();
    expect(screen.getByText("₹4.35")).toBeInTheDocument();
    expect(screen.getByText("Average Cost")).toBeInTheDocument();
    expect(screen.getByText("₹2.17")).toBeInTheDocument();
  });

  it("shows an empty state when there are no generations", async () => {
    generationHistoryApi.list.mockResolvedValue({ items: [] });
    generationHistoryApi.stats.mockResolvedValue({ ...SAMPLE_STATS, totalGenerations: 0 });

    renderPage();

    expect(await screen.findByText(/no generations yet/i)).toBeInTheDocument();
  });

  it("shows an error state when the API call fails", async () => {
    generationHistoryApi.list.mockRejectedValue(new Error("API request failed: 500 Internal Server Error"));
    generationHistoryApi.stats.mockResolvedValue(SAMPLE_STATS);

    renderPage();

    expect(await screen.findByText(/couldn't load test plan generation history/i)).toBeInTheDocument();
  });

  it("opens the detail drawer with generation details when View is clicked", async () => {
    const user = userEvent.setup();
    generationHistoryApi.list.mockResolvedValue({ items: [SAMPLE_SUMMARY] });
    generationHistoryApi.stats.mockResolvedValue(SAMPLE_STATS);
    generationHistoryApi.getById.mockResolvedValue(SAMPLE_DETAIL);

    renderPage();
    await screen.findByText("Contact Log");

    await user.click(screen.getByRole("button", { name: "View" }));

    expect(generationHistoryApi.getById).toHaveBeenCalledWith("2026-08-03T10-00-00");

    const drawer = await screen.findByRole("dialog", { name: "Generation Details" });
    expect(within(drawer).getByText("Log a new contact")).toBeInTheDocument();
    expect(within(drawer).getByText("Search existing contacts")).toBeInTheDocument();
    expect(within(drawer).getByText("54980")).toBeInTheDocument();

    // The Generation Query toolbar appears first, above metadata/usage/
    // cost/retrieval/test cases, and recovers the original request from
    // the persisted prompt text without any extra API call.
    expect(within(drawer).getByText("Generation Query")).toBeInTheDocument();
    expect(within(drawer).getByText(/Optum requested a Bridge indicator/)).toBeInTheDocument();
    expect(within(drawer).getByText(/Generate regression test cases/)).toBeInTheDocument();
    expect(generationHistoryApi.getById).toHaveBeenCalledTimes(1);
  });

  it("closes the detail drawer when the close button is clicked", async () => {
    const user = userEvent.setup();
    generationHistoryApi.list.mockResolvedValue({ items: [SAMPLE_SUMMARY] });
    generationHistoryApi.stats.mockResolvedValue(SAMPLE_STATS);
    generationHistoryApi.getById.mockResolvedValue(SAMPLE_DETAIL);

    renderPage();
    await screen.findByText("Contact Log");
    await user.click(screen.getByRole("button", { name: "View" }));
    await screen.findByRole("dialog", { name: "Generation Details" });

    await user.click(screen.getByRole("button", { name: /close/i }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "Generation Details" })).not.toBeInTheDocument();
    });
  });

  it("downloads the Excel report when Download Excel is clicked", async () => {
    const user = userEvent.setup();
    generationHistoryApi.list.mockResolvedValue({ items: [SAMPLE_SUMMARY] });
    generationHistoryApi.stats.mockResolvedValue(SAMPLE_STATS);
    generationHistoryApi.downloadExcel.mockResolvedValue(undefined);

    renderPage();
    await screen.findByText("Contact Log");

    await user.click(screen.getByRole("button", { name: "Download Excel" }));

    expect(generationHistoryApi.downloadExcel).toHaveBeenCalledWith("2026-08-03T10-00-00");
    expect(generationHistoryApi.getById).not.toHaveBeenCalled();
  });

  it("shows an error message if the Excel download fails", async () => {
    const user = userEvent.setup();
    generationHistoryApi.list.mockResolvedValue({ items: [SAMPLE_SUMMARY] });
    generationHistoryApi.stats.mockResolvedValue(SAMPLE_STATS);
    generationHistoryApi.downloadExcel.mockRejectedValue(new Error("API request failed: 404 Not Found"));

    renderPage();
    await screen.findByText("Contact Log");

    await user.click(screen.getByRole("button", { name: "Download Excel" }));

    expect(await screen.findByText(/API request failed: 404/i)).toBeInTheDocument();
  });
});
