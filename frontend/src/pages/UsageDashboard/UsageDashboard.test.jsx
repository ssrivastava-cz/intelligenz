import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { indexingHistoryApi } from "../../api/indexingHistoryApi.js";
import { usageApi } from "../../api/usageApi.js";
import UsageDashboard from "./UsageDashboard.jsx";

vi.mock("../../api/usageApi.js", () => ({
  usageApi: {
    get: vi.fn(),
  },
}));

vi.mock("../../api/indexingHistoryApi.js", () => ({
  indexingHistoryApi: {
    list: vi.fn(),
    stats: vi.fn(),
  },
}));

// A Test Plan Generator generation, normalized.
const TEST_PLAN_GENERATOR_RECORD = {
  generationId: "2026-08-03T10-00-00",
  source: "test_plan_generator",
  model: "gpt-5",
  inputTokens: 22128,
  outputTokens: 7615,
  totalTokens: 29743,
  estimatedInputTokens: 21000,
  inputCostInr: 5.87,
  outputCostInr: 4.86,
  totalCostInr: 10.73,
  generationTimeMs: 7200,
  createdAt: "2026-08-03T10:00:00Z",
};

// A Knowledge Assistant generation, normalized from usage.json.
const KNOWLEDGE_ASSISTANT_RECORD = {
  generationId: "gen_20260802_abc123",
  source: "knowledge_assistant",
  model: "gpt-5",
  inputTokens: 3200,
  outputTokens: 580,
  totalTokens: 3780,
  estimatedInputTokens: 3500,
  inputCostInr: 0.4,
  outputCostInr: 0.55,
  totalCostInr: 0.95,
  generationTimeMs: 8420,
  createdAt: "2026-08-02T09:00:00Z",
};

const SAMPLE_SUMMARY = {
  totalGenerations: 2,
  totalTokens: TEST_PLAN_GENERATOR_RECORD.totalTokens + KNOWLEDGE_ASSISTANT_RECORD.totalTokens,
  totalAiCostInr: TEST_PLAN_GENERATOR_RECORD.totalCostInr + KNOWLEDGE_ASSISTANT_RECORD.totalCostInr,
  averageCostInr: (TEST_PLAN_GENERATOR_RECORD.totalCostInr + KNOWLEDGE_ASSISTANT_RECORD.totalCostInr) / 2,
};

const EMPTY_SUMMARY = { totalGenerations: 0, totalTokens: 0, totalAiCostInr: 0, averageCostInr: 0 };

const INDEXING_RUN = {
  activityType: "DOCUMENT_INDEXING",
  historyId: "2026-08-02_19-22-16",
  feature: "Contact and Sticket Log",
  indexedAt: "2026-08-02T19:22:16.291076Z",
  embeddingModel: "text-embedding-3-small",
  documentsIndexed: 6,
  chunksIndexed: 161,
  embeddingTokens: 22073,
  averageTokensPerChunk: 137.09937888198758,
  embeddingCostUsd: 0.00044146,
  embeddingCostInr: 0.04,
  elapsedSeconds: 6.80998000013642,
  chromaCollectionName: "source_of_truth_chunks",
  status: "SUCCESS",
};

const EMPTY_INDEXING_STATS = {
  totalIndexingRuns: 0,
  totalDocumentsIndexed: 0,
  totalChunksIndexed: 0,
  totalEmbeddingTokens: 0,
  totalEmbeddingCostUsd: 0.0,
  totalEmbeddingCostInr: 0.0,
};

const INDEXING_STATS = {
  totalIndexingRuns: 1,
  totalDocumentsIndexed: 6,
  totalChunksIndexed: 161,
  totalEmbeddingTokens: 22073,
  totalEmbeddingCostUsd: 0.00044146,
  totalEmbeddingCostInr: 0.04,
};

function renderPage() {
  return render(<UsageDashboard />);
}

describe("UsageDashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Most tests only care about the Generation side — default the
    // Indexing side to "nothing indexed yet" so those assertions don't
    // need to know about it too.
    indexingHistoryApi.list.mockResolvedValue([]);
    indexingHistoryApi.stats.mockResolvedValue(EMPTY_INDEXING_STATS);
  });

  // --- 1. loading state ---

  it("shows a loading state while data is being fetched", () => {
    usageApi.get.mockReturnValue(new Promise(() => {}));

    renderPage();

    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  // --- 2. empty state ---

  it("shows the empty state when there is no usage and no indexing history", async () => {
    usageApi.get.mockResolvedValue({ summary: EMPTY_SUMMARY, generations: [] });

    renderPage();

    expect(await screen.findByText("No AI usage available yet.")).toBeInTheDocument();
    expect(screen.getByText("Generate your first AI test plan to begin tracking usage.")).toBeInTheDocument();
  });

  // --- API error handling ---

  it("shows the error state when the usage request fails", async () => {
    usageApi.get.mockRejectedValue(new Error("API request failed: 500 Internal Server Error"));

    renderPage();

    expect(await screen.findByText("Unable to load usage information.")).toBeInTheDocument();
    expect(screen.getByText("Please try again.")).toBeInTheDocument();
  });

  it("shows the error state (cleanly, without crashing) when the indexing request fails", async () => {
    usageApi.get.mockResolvedValue({ summary: SAMPLE_SUMMARY, generations: [TEST_PLAN_GENERATOR_RECORD] });
    indexingHistoryApi.list.mockRejectedValue(new Error("API request failed: 500 Internal Server Error"));

    renderPage();

    expect(await screen.findByText("Unable to load usage information.")).toBeInTheDocument();
  });

  // --- 3. combined summary metrics ---

  it("renders the combined summary cards across both sources", async () => {
    usageApi.get.mockResolvedValue({
      summary: SAMPLE_SUMMARY,
      generations: [TEST_PLAN_GENERATOR_RECORD, KNOWLEDGE_ASSISTANT_RECORD],
    });

    const { container } = renderPage();
    await screen.findByText("AI Generation Usage");

    const statsSection = container.querySelector(".usage-dashboard__stats");
    expect(within(statsSection).getByText("Total Generations")).toBeInTheDocument();
    expect(within(statsSection).getByText("2")).toBeInTheDocument();
    expect(within(statsSection).getByText("Total Tokens")).toBeInTheDocument();
    expect(within(statsSection).getByText(SAMPLE_SUMMARY.totalTokens.toLocaleString())).toBeInTheDocument();
    expect(within(statsSection).getByText("Total AI Cost (INR)")).toBeInTheDocument();
    expect(within(statsSection).getByText("₹11.68")).toBeInTheDocument();
    expect(within(statsSection).getByText("Average Cost Per Generation")).toBeInTheDocument();
    expect(within(statsSection).getByText("₹5.84")).toBeInTheDocument();
  });

  // --- 4./5. both source types render + Source column/badge ---

  it("renders both a Test Plan Generator row and a Knowledge Assistant row, each tagged with its Source", async () => {
    usageApi.get.mockResolvedValue({
      summary: SAMPLE_SUMMARY,
      generations: [TEST_PLAN_GENERATOR_RECORD, KNOWLEDGE_ASSISTANT_RECORD],
    });

    renderPage();
    const section = (await screen.findByText("AI Generation Usage")).closest("section");

    expect(within(section).getByText("2026-08-03T10-00-00")).toBeInTheDocument();
    expect(within(section).getByText("gen_20260802_abc123")).toBeInTheDocument();
    expect(within(section).getByText("Test Plan Generator")).toBeInTheDocument();
    expect(within(section).getByText("Knowledge Assistant")).toBeInTheDocument();
  });

  it("renders every unified usage column for a Knowledge Assistant row", async () => {
    usageApi.get.mockResolvedValue({ summary: SAMPLE_SUMMARY, generations: [KNOWLEDGE_ASSISTANT_RECORD] });

    renderPage();
    const section = (await screen.findByText("AI Generation Usage")).closest("section");

    expect(within(section).getByText("gen_20260802_abc123")).toBeInTheDocument();
    expect(within(section).getByText("Knowledge Assistant")).toBeInTheDocument();
    expect(within(section).getByText("3,500")).toBeInTheDocument(); // estimated input tokens
    expect(within(section).getByText("3,200")).toBeInTheDocument(); // actual input tokens
    expect(within(section).getByText("580")).toBeInTheDocument(); // output tokens
    expect(within(section).getByText("3,780")).toBeInTheDocument(); // total tokens
    expect(within(section).getByText("₹0.40")).toBeInTheDocument(); // input cost
    expect(within(section).getByText("₹0.55")).toBeInTheDocument(); // output cost
    expect(within(section).getByText("₹0.95")).toBeInTheDocument(); // total AI cost
    expect(within(section).getByText("8.4 sec")).toBeInTheDocument(); // generation time
  });

  it("renders reasoning/visible-answer/structured-output-overhead tokens for a Knowledge Assistant row", async () => {
    const recordWithTokenBreakdown = {
      ...KNOWLEDGE_ASSISTANT_RECORD,
      outputTokens: 1094,
      reasoningTokens: 960,
      visibleAnswerTokens: 78,
      structuredOutputOverheadTokens: 56,
    };
    usageApi.get.mockResolvedValue({ summary: SAMPLE_SUMMARY, generations: [recordWithTokenBreakdown] });

    renderPage();
    const section = (await screen.findByText("AI Generation Usage")).closest("section");

    expect(within(section).getByText("1,094")).toBeInTheDocument(); // billed output tokens
    expect(within(section).getByText("960")).toBeInTheDocument(); // reasoning tokens
    expect(within(section).getByText("78")).toBeInTheDocument(); // visible answer tokens
    expect(within(section).getByText("56")).toBeInTheDocument(); // structured output overhead
  });

  it("shows a placeholder for reasoning/visible-answer tokens on a Test Plan Generator row (not tracked for that source)", async () => {
    usageApi.get.mockResolvedValue({ summary: SAMPLE_SUMMARY, generations: [TEST_PLAN_GENERATOR_RECORD] });

    renderPage();
    const section = (await screen.findByText("AI Generation Usage")).closest("section");
    const [row] = within(section).getAllByRole("row").slice(1);

    // reasoningTokens, visibleAnswerTokens, structuredOutputOverheadTokens
    // are all undefined on a Test Plan Generator record.
    expect(within(row).getAllByText("—").length).toBeGreaterThanOrEqual(3);
  });

  it("never calls indexingHistoryApi more than once (no per-row detail fetches)", async () => {
    usageApi.get.mockResolvedValue({ summary: SAMPLE_SUMMARY, generations: [TEST_PLAN_GENERATOR_RECORD] });

    renderPage();
    await screen.findByText("AI Generation Usage");

    expect(usageApi.get).toHaveBeenCalledTimes(1);
    expect(indexingHistoryApi.list).toHaveBeenCalledTimes(1);
    expect(indexingHistoryApi.stats).toHaveBeenCalledTimes(1);
  });

  // --- Document Indexing & Embedding (unchanged section, still exercised) ---

  it("renders the indexing summary cards and table alongside the unified usage table", async () => {
    usageApi.get.mockResolvedValue({ summary: SAMPLE_SUMMARY, generations: [TEST_PLAN_GENERATOR_RECORD] });
    indexingHistoryApi.list.mockResolvedValue([INDEXING_RUN]);
    indexingHistoryApi.stats.mockResolvedValue(INDEXING_STATS);

    renderPage();
    const section = (await screen.findByText("Document Indexing & Embedding")).closest("section");

    expect(within(section).getByText("Contact and Sticket Log")).toBeInTheDocument();
    expect(within(section).getByText("text-embedding-3-small")).toBeInTheDocument();
    expect(within(section).getByText("₹0.04")).toBeInTheDocument();
    // The generation side is unaffected by the indexing side rendering.
    expect(screen.getByText("AI Generation Usage")).toBeInTheDocument();
    expect(screen.getByText("2026-08-03T10-00-00")).toBeInTheDocument();
  });

  it("shows a clean empty state for indexing history while the unified usage table still renders", async () => {
    usageApi.get.mockResolvedValue({ summary: SAMPLE_SUMMARY, generations: [TEST_PLAN_GENERATOR_RECORD] });
    // indexingHistoryApi already defaults to empty via beforeEach.

    renderPage();

    expect(await screen.findByText("No indexing runs yet.")).toBeInTheDocument();
    expect(screen.queryByText("Document Indexing & Embedding")).not.toBeInTheDocument();
    expect(screen.getByText("AI Generation Usage")).toBeInTheDocument();
  });

  it("shows a clean empty state for usage while the indexing table still renders", async () => {
    usageApi.get.mockResolvedValue({ summary: EMPTY_SUMMARY, generations: [] });
    indexingHistoryApi.list.mockResolvedValue([INDEXING_RUN]);
    indexingHistoryApi.stats.mockResolvedValue(INDEXING_STATS);

    renderPage();

    expect(await screen.findByText("No AI generations yet.")).toBeInTheDocument();
    expect(screen.queryByText("AI Generation Usage")).not.toBeInTheDocument();
    expect(screen.getByText("Document Indexing & Embedding")).toBeInTheDocument();
  });
});
