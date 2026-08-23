import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { generationApi } from "../../api/generationApi.js";
import { generationHistoryApi } from "../../api/generationHistoryApi.js";
import { redmineApi } from "../../api/redmineApi.js";
import { sourceOfTruthApi } from "../../api/sourceOfTruthApi.js";
import { uploadApi } from "../../api/uploadApi.js";
import TestPlanGenerator from "./TestPlanGenerator.jsx";

vi.mock("../../api/sourceOfTruthApi.js", () => ({ sourceOfTruthApi: { listFeatures: vi.fn() } }));
vi.mock("../../api/uploadApi.js", () => ({
  uploadApi: { uploadFiles: vi.fn(), getEmbeddingPreview: vi.fn(), embedDocuments: vi.fn() },
}));
vi.mock("../../api/generationApi.js", () => ({ generationApi: { generate: vi.fn(), getProgress: vi.fn() } }));
vi.mock("../../api/generationHistoryApi.js", () => ({
  generationHistoryApi: { getById: vi.fn(), downloadExcel: vi.fn() },
}));
vi.mock("../../api/redmineApi.js", () => ({ redmineApi: { getTicketDetail: vi.fn() } }));

const FEATURES = [{ feature: "Appointments", documents: 3, workflows: 1, testCases: 1, issues: 1 }];

const UPLOAD_RESPONSE = {
  uploadSessionId: "sess-1",
  documentsUploaded: 1,
  uploadedFiles: [{ documentId: "doc-1", documentName: "notes.txt" }],
};

const PREVIEW_RESPONSE = {
  uploadSessionId: "sess-1",
  documentsFound: 1,
  chunksCreated: 3,
  embeddingModel: "text-embedding-3-small",
  totalEmbeddingTokens: 120,
  averageTokensPerChunk: 40,
  estimatedEmbeddingCost: 0.0001,
  estimatedEmbeddingCostInr: 0.01,
  chunks: [],
};

const EMBED_RESPONSE = {
  uploadSessionId: "sess-1",
  uploadedAt: "2026-08-06T00:00:00Z",
  embeddingModel: "text-embedding-3-small",
  documentsIndexed: 1,
  chunksIndexed: 3,
  averageTokensPerChunk: 40,
  estimatedEmbeddingCost: 0.0001,
  elapsedSeconds: 1.2,
  chromaCollectionName: "uploaded_documents_sess-1",
  indexStatus: "SUCCESS",
};

const TICKET_DETAIL = { ticketId: "12345", description: "Users should be able to reschedule appointments." };
const GENERATE_RESPONSE = { generationId: "gen-1" };
const GENERATION_DETAIL = {
  generationId: "gen-1",
  generationTimeMs: 7200,
  actualUsage: { promptTokens: 1000, completionTokens: 200, totalTokens: 1200, totalCost: { usd: 0.01, inr: 0.87 } },
  estimatedUsage: { estimatedInputCost: { usd: 0.005, inr: 0.44 } },
  testCases: [
    {
      requirementId: "REQ-1",
      testCaseId: "TC-1",
      testCaseTitle: "Reschedule an appointment",
      priority: "High",
      testSuite: "Appointments",
      preconditions: "User is logged in.",
      steps: [{ stepNo: 1, action: "Open appointment.", expectedResult: "Details are shown." }],
      postConditions: "Appointment is updated.",
      automationStatus: "Not Automated",
      testType: "Functional",
      tags: ["Appointments"],
    },
  ],
};

function setUpHappyPath() {
  sourceOfTruthApi.listFeatures.mockResolvedValue(FEATURES);
  uploadApi.uploadFiles.mockResolvedValue(UPLOAD_RESPONSE);
  uploadApi.getEmbeddingPreview.mockResolvedValue(PREVIEW_RESPONSE);
  uploadApi.embedDocuments.mockResolvedValue(EMBED_RESPONSE);
  redmineApi.getTicketDetail.mockResolvedValue(TICKET_DETAIL);
  generationApi.generate.mockResolvedValue(GENERATE_RESPONSE);
  generationApi.getProgress.mockResolvedValue({ stage: null });
  generationHistoryApi.getById.mockResolvedValue(GENERATION_DETAIL);
  generationHistoryApi.downloadExcel.mockResolvedValue(undefined);
}

async function fillFeatureAndTicket(user) {
  await screen.findByRole("option", { name: "Appointments" });
  await user.selectOptions(screen.getByLabelText(/feature/i), "Appointments");
  await user.type(screen.getByLabelText(/redmine ticket id/i), "12345");
}

async function fillFeatureOnly(user) {
  await screen.findByRole("option", { name: "Appointments" });
  await user.selectOptions(screen.getByLabelText(/feature/i), "Appointments");
}

describe("TestPlanGenerator", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("loads the real feature list into the Feature dropdown", async () => {
    sourceOfTruthApi.listFeatures.mockResolvedValue(FEATURES);

    render(<TestPlanGenerator />);

    expect(await screen.findByRole("option", { name: "Appointments" })).toBeInTheDocument();
  });

  it("uploads a selected file, then automatically shows the embedding preview", async () => {
    const user = userEvent.setup();
    setUpHappyPath();
    render(<TestPlanGenerator />);
    await fillFeatureAndTicket(user);

    const file = new File(["content"], "notes.txt", { type: "text/plain" });
    const fileInput = document.querySelector('input[type="file"]');
    await user.upload(fileInput, file);

    expect(await screen.findByText(/1 document uploaded/i)).toBeInTheDocument();
    expect(uploadApi.uploadFiles).toHaveBeenCalledWith([file]);

    expect(await screen.findByText("Documents Found")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument(); // chunks created
    expect(screen.getByText(/\$0\.0001/)).toBeInTheDocument();
    expect(screen.getByText(/₹0\.01/)).toBeInTheDocument();
  });

  it("shows an upload error with a working retry action", async () => {
    const user = userEvent.setup();
    sourceOfTruthApi.listFeatures.mockResolvedValue(FEATURES);
    uploadApi.uploadFiles.mockRejectedValueOnce(new Error("Unable to reach the server."));
    render(<TestPlanGenerator />);
    await fillFeatureAndTicket(user);

    const file = new File(["content"], "notes.txt", { type: "text/plain" });
    await user.upload(document.querySelector('input[type="file"]'), file);

    expect(await screen.findByText("Unable to reach the server.")).toBeInTheDocument();

    uploadApi.uploadFiles.mockResolvedValue(UPLOAD_RESPONSE);
    uploadApi.getEmbeddingPreview.mockResolvedValue(PREVIEW_RESPONSE);
    await user.click(screen.getByRole("button", { name: /retry/i }));

    expect(await screen.findByText(/1 document uploaded/i)).toBeInTheDocument();
  });

  it("keeps Generate disabled until an uploaded session is embedded, then enables it after Confirm & Embed", async () => {
    const user = userEvent.setup();
    setUpHappyPath();
    render(<TestPlanGenerator />);
    await fillFeatureAndTicket(user);

    const generateButton = screen.getByRole("button", { name: /generate test cases/i });
    expect(generateButton).toBeEnabled();

    const file = new File(["content"], "notes.txt", { type: "text/plain" });
    await user.upload(document.querySelector('input[type="file"]'), file);
    await screen.findByText("Documents Found");

    expect(screen.getByRole("button", { name: /generate test cases/i })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: /confirm & embed documents/i }));

    expect(await screen.findByText(/embedding complete — 3 chunks embedded/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /generate test cases/i })).toBeEnabled();
  });

  it("shows a generating indicator while the request is in flight", async () => {
    const user = userEvent.setup();
    sourceOfTruthApi.listFeatures.mockResolvedValue(FEATURES);
    redmineApi.getTicketDetail.mockReturnValue(new Promise(() => {}));
    render(<TestPlanGenerator />);
    await fillFeatureAndTicket(user);

    await user.click(screen.getByRole("button", { name: /generate test cases/i }));

    expect(await screen.findByText(/fetching ticket details/i)).toBeInTheDocument();
  });

  it("runs the full generate workflow and displays the summary, results, and enables download", async () => {
    const user = userEvent.setup();
    setUpHappyPath();
    render(<TestPlanGenerator />);
    await fillFeatureAndTicket(user);

    await user.click(screen.getByRole("button", { name: /generate test cases/i }));

    await waitFor(() =>
      expect(generationApi.generate).toHaveBeenCalledWith(
        expect.objectContaining({
          feature: "Appointments",
          redmineId: "12345",
          redmineDescription: TICKET_DETAIL.description,
          optionalDescription: undefined,
          uploadSessionId: undefined,
          requestId: expect.any(String),
        }),
      ),
    );

    expect(await screen.findByText("Generation Summary")).toBeInTheDocument();
    expect(screen.getByText("7.2 sec")).toBeInTheDocument();
    expect(screen.getByText("1,000")).toBeInTheDocument();
    expect(screen.getByText("200")).toBeInTheDocument();
    expect(screen.getByText("1,200")).toBeInTheDocument();

    expect(screen.getByText("Reschedule an appointment")).toBeInTheDocument();

    const downloadButton = screen.getByRole("button", { name: /download excel/i });
    await user.click(downloadButton);
    expect(generationHistoryApi.downloadExcel).toHaveBeenCalledWith("gen-1");
  });

  it("renders every returned test case, never silently slicing the results (e.g. down to 5)", async () => {
    const user = userEvent.setup();
    setUpHappyPath();
    const manyTestCases = Array.from({ length: 9 }, (_, index) => ({
      ...GENERATION_DETAIL.testCases[0],
      testCaseId: `TC-${index + 1}`,
      testCaseTitle: `Generated scenario ${index + 1}`,
    }));
    generationHistoryApi.getById.mockResolvedValue({ ...GENERATION_DETAIL, testCases: manyTestCases });
    render(<TestPlanGenerator />);
    await fillFeatureAndTicket(user);

    await user.click(screen.getByRole("button", { name: /generate test cases/i }));

    expect(await screen.findByText("Generation Summary")).toBeInTheDocument();
    for (const testCase of manyTestCases) {
      expect(screen.getByText(testCase.testCaseTitle)).toBeInTheDocument();
    }
    // Header row + one row per test case — none dropped.
    expect(screen.getAllByRole("row")).toHaveLength(manyTestCases.length + 1);
  });

  it("generates successfully from the Feature alone, with no Redmine Ticket ID entered", async () => {
    const user = userEvent.setup();
    setUpHappyPath();
    render(<TestPlanGenerator />);
    await fillFeatureOnly(user);

    expect(screen.getByRole("button", { name: /generate test cases/i })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /generate test cases/i }));

    await waitFor(() =>
      expect(generationApi.generate).toHaveBeenCalledWith(
        expect.objectContaining({
          feature: "Appointments",
          redmineId: undefined,
          redmineDescription: undefined,
        }),
      ),
    );
    expect(redmineApi.getTicketDetail).not.toHaveBeenCalled();
    expect(await screen.findByText("Generation Summary")).toBeInTheDocument();
    expect(screen.getByText("Reschedule an appointment")).toBeInTheDocument();
  });

  it("shows the real backend stage list, without a Redmine step, when no ticket id was given", async () => {
    const user = userEvent.setup();
    sourceOfTruthApi.listFeatures.mockResolvedValue(FEATURES);
    generationApi.generate.mockReturnValue(new Promise(() => {}));
    render(<TestPlanGenerator />);
    await fillFeatureOnly(user);

    await user.click(screen.getByRole("button", { name: /generate test cases/i }));

    expect(await screen.findByText("Retrieving relevant documents")).toBeInTheDocument();
    expect(screen.queryByText("Fetching Redmine ticket details")).not.toBeInTheDocument();
  });

  it("passes the confirmed upload session id into the generate request once embedded", async () => {
    const user = userEvent.setup();
    setUpHappyPath();
    render(<TestPlanGenerator />);
    await fillFeatureAndTicket(user);

    const file = new File(["content"], "notes.txt", { type: "text/plain" });
    await user.upload(document.querySelector('input[type="file"]'), file);
    await screen.findByText("Documents Found");
    await user.click(screen.getByRole("button", { name: /confirm & embed documents/i }));
    await screen.findByText(/embedding complete/i);

    await user.click(screen.getByRole("button", { name: /generate test cases/i }));

    await waitFor(() =>
      expect(generationApi.generate).toHaveBeenCalledWith(expect.objectContaining({ uploadSessionId: "sess-1" })),
    );
  });

  it("shows a generation error when the Redmine ticket cannot be found", async () => {
    const user = userEvent.setup();
    sourceOfTruthApi.listFeatures.mockResolvedValue(FEATURES);
    redmineApi.getTicketDetail.mockRejectedValue(new Error("No Redmine ticket found with id '12345'."));
    render(<TestPlanGenerator />);
    await fillFeatureAndTicket(user);

    await user.click(screen.getByRole("button", { name: /generate test cases/i }));

    expect(await screen.findByText("Generation failed")).toBeInTheDocument();
    expect(screen.getByText("No Redmine ticket found with id '12345'.")).toBeInTheDocument();
    expect(generationApi.generate).not.toHaveBeenCalled();
  });

  it("shows a generation error when the AI generation call fails", async () => {
    const user = userEvent.setup();
    sourceOfTruthApi.listFeatures.mockResolvedValue(FEATURES);
    redmineApi.getTicketDetail.mockResolvedValue(TICKET_DETAIL);
    generationApi.generate.mockRejectedValue(new Error("OpenAI chat completion request failed."));
    render(<TestPlanGenerator />);
    await fillFeatureAndTicket(user);

    await user.click(screen.getByRole("button", { name: /generate test cases/i }));

    expect(await screen.findByText("Generation failed")).toBeInTheDocument();
    expect(screen.getByText("OpenAI chat completion request failed.")).toBeInTheDocument();
  });

  it("shows a download error inline without discarding the displayed results", async () => {
    const user = userEvent.setup();
    setUpHappyPath();
    generationHistoryApi.downloadExcel.mockRejectedValue(new Error("The requested resource could not be found."));
    render(<TestPlanGenerator />);
    await fillFeatureAndTicket(user);

    await user.click(screen.getByRole("button", { name: /generate test cases/i }));
    await screen.findByText("Generation Summary");

    await user.click(screen.getByRole("button", { name: /download excel/i }));

    expect(await screen.findByText("The requested resource could not be found.")).toBeInTheDocument();
    expect(screen.getByText("Reschedule an appointment")).toBeInTheDocument();
  });
});
