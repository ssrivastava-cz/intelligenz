import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { knowledgeAssistantApi } from "../../api/knowledgeAssistantApi.js";
import { sourceOfTruthApi } from "../../api/sourceOfTruthApi.js";
import KnowledgeAssistant from "./KnowledgeAssistant.jsx";

vi.mock("../../api/knowledgeAssistantApi.js", () => ({
  knowledgeAssistantApi: {
    ask: vi.fn(),
    submitFeedback: vi.fn(),
    getDebug: vi.fn(),
    listRecent: vi.fn(),
  },
}));

vi.mock("../../api/sourceOfTruthApi.js", () => ({
  sourceOfTruthApi: { listFeatures: vi.fn() },
}));

const FEATURES = [{ feature: "Contact Log", documents: 6, workflows: 2, testCases: 3, issues: 1 }];

const RECENT_GENERATIONS = [
  {
    generationId: "gen_20260810_real1",
    question: "How is the AI cost calculated for a test plan generation?",
    answer: "Cost is calculated based on the tokens used during prompt construction and AI generation.",
    sourceDocuments: ["Usage & Billing Guide.pdf"],
    createdAt: "2026-08-10T21:20:00Z",
  },
  {
    generationId: "gen_20260809_real2",
    question: "Where can I see the token usage for a specific test plan run?",
    answer: "Open the generation's entry in Test Plan Generation History and click View.",
    sourceDocuments: [],
    createdAt: "2026-08-09T09:00:00Z",
  },
];

const ASK_RESPONSE = {
  generationId: "gen_20260812_abc123",
  question: "How does Contact Log handle bridged contacts?",
  answer: "According to the workflow, bridged contacts are merged automatically.",
  sourceDocuments: ["Contact_Log_Workflow.pdf", "Contact_Log_TestCases.xlsx"],
  usage: {
    inputTokens: 3200,
    outputTokens: 580,
    totalTokens: 3780,
    estimatedInputTokens: 3500,
    inputCostInr: 0.4,
    outputCostInr: 0.55,
    totalCostInr: 0.95,
  },
};

function renderPage() {
  return render(<KnowledgeAssistant />);
}

async function selectFeature(user, feature = "Contact Log") {
  await screen.findByRole("option", { name: feature });
  await user.selectOptions(screen.getByLabelText(/feature/i), feature);
}

async function askQuestion(user, question = "How does Contact Log handle bridged contacts?") {
  await selectFeature(user);
  await user.type(
    screen.getByPlaceholderText("e.g. How is the AI cost calculated for a test plan generation?"),
    question,
  );
  await user.click(screen.getByRole("button", { name: "Ask" }));
}

describe("KnowledgeAssistant page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sourceOfTruthApi.listFeatures.mockResolvedValue(FEATURES);
    knowledgeAssistantApi.ask.mockResolvedValue(ASK_RESPONSE);
    knowledgeAssistantApi.submitFeedback.mockResolvedValue({
      generationId: ASK_RESPONSE.generationId,
      evaluation: "GOOD",
    });
    knowledgeAssistantApi.listRecent.mockResolvedValue(RECENT_GENERATIONS);
  });

  it("renders the Ask a question card with a real feature dropdown", async () => {
    renderPage();

    expect(screen.getByText("Ask a question")).toBeInTheDocument();
    expect(screen.getByText("Search across product docs, policies, and internal guides.")).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText("e.g. How is the AI cost calculated for a test plan generation?"),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ask" })).toBeInTheDocument();
    expect(await screen.findByRole("option", { name: "Contact Log" })).toBeInTheDocument();
  });

  it("fetches the real recent generations from the backend and shows them collapsed by default", async () => {
    renderPage();

    expect(await screen.findByText(`${RECENT_GENERATIONS.length} questions asked recently`)).toBeInTheDocument();
    expect(knowledgeAssistantApi.listRecent).toHaveBeenCalled();
    expect(screen.queryByText(RECENT_GENERATIONS[0].question)).not.toBeInTheDocument();
  });

  it("expands the recent questions section, then expands one item to show its real persisted answer and documents", async () => {
    const user = userEvent.setup();
    renderPage();
    await screen.findByText(`${RECENT_GENERATIONS.length} questions asked recently`);

    await user.click(screen.getByRole("button", { name: /expand recent questions/i }));
    const [firstRecent] = RECENT_GENERATIONS;
    expect(screen.getByText(firstRecent.question)).toBeInTheDocument();
    expect(screen.queryByText(firstRecent.answer)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: new RegExp(firstRecent.question) }));

    expect(screen.getByText(firstRecent.answer)).toBeInTheDocument();
    expect(screen.getByText(firstRecent.sourceDocuments[0])).toBeInTheDocument();
  });

  it("shows a clean empty state, never mock questions, when the backend history fetch fails", async () => {
    knowledgeAssistantApi.listRecent.mockRejectedValue(new Error("The server encountered an error."));
    renderPage();

    expect(await screen.findByText("0 questions asked recently")).toBeInTheDocument();
    expect(screen.queryByText(RECENT_GENERATIONS[0].question)).not.toBeInTheDocument();
  });

  // --- Ask: real API, not the mock ---

  it("sends the question and selected feature to the real backend, not the mock service", async () => {
    const user = userEvent.setup();
    renderPage();

    await askQuestion(user);

    expect(knowledgeAssistantApi.ask).toHaveBeenCalledWith({
      question: "How does Contact Log handle bridged contacts?",
      feature: "Contact Log",
    });
  });

  it("renders the backend's answer, question, and generationId-linked record after asking", async () => {
    const user = userEvent.setup();
    renderPage();

    await askQuestion(user);

    expect(await screen.findByText("Question")).toBeInTheDocument();
    expect(screen.getByText("How does Contact Log handle bridged contacts?")).toBeInTheDocument();
    expect(screen.getByText("Answer")).toBeInTheDocument();
    expect(screen.getByText("According to the workflow, bridged contacts are merged automatically.")).toBeInTheDocument();
  });

  it("renders every source document the backend returned", async () => {
    const user = userEvent.setup();
    renderPage();

    await askQuestion(user);

    await screen.findByText("Documents used in this answer");
    expect(screen.getByText("Contact_Log_Workflow.pdf")).toBeInTheDocument();
    expect(screen.getByText("Contact_Log_TestCases.xlsx")).toBeInTheDocument();
  });

  it("adds a successful question to the recent questions list", async () => {
    const user = userEvent.setup();
    renderPage();
    await screen.findByText(`${RECENT_GENERATIONS.length} questions asked recently`);

    await askQuestion(user);

    await screen.findByText("Question");
    expect(screen.getByText(`${RECENT_GENERATIONS.length + 1} questions asked recently`)).toBeInTheDocument();
  });

  it("disables the Ask button while the request is in flight, and re-enables it afterward", async () => {
    const user = userEvent.setup();
    let resolveAsk;
    knowledgeAssistantApi.ask.mockReturnValue(
      new Promise((resolve) => {
        resolveAsk = resolve;
      }),
    );
    renderPage();
    await selectFeature(user);
    await user.type(
      screen.getByPlaceholderText("e.g. How is the AI cost calculated for a test plan generation?"),
      "How does Contact Log work?",
    );

    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(screen.getByRole("button", { name: "Asking…" })).toBeDisabled();
    expect(knowledgeAssistantApi.ask).toHaveBeenCalledTimes(1);

    resolveAsk(ASK_RESPONSE);
    await screen.findByRole("button", { name: "Ask" });
    expect(screen.getByRole("button", { name: "Ask" })).toBeEnabled();
  });

  // --- Errors ---

  it("shows the existing error UI when the backend returns an error, without a fake answer", async () => {
    const user = userEvent.setup();
    knowledgeAssistantApi.ask.mockRejectedValue(new Error("The server encountered an error. Please try again in a moment."));
    renderPage();

    await askQuestion(user);

    expect(await screen.findByText("Couldn't get an answer")).toBeInTheDocument();
    expect(
      screen.getByText("The server encountered an error. Please try again in a moment."),
    ).toBeInTheDocument();
    expect(screen.queryByText("Question")).not.toBeInTheDocument();
  });

  it("clears the loading state after an error, so Ask is usable again", async () => {
    const user = userEvent.setup();
    knowledgeAssistantApi.ask.mockRejectedValue(new Error("Something went wrong."));
    renderPage();

    await askQuestion(user);

    await screen.findByText("Couldn't get an answer");
    expect(screen.getByRole("button", { name: "Ask" })).toBeEnabled();
  });

  it("does not add a failed question to the recent questions list", async () => {
    const user = userEvent.setup();
    knowledgeAssistantApi.ask.mockRejectedValue(new Error("Something went wrong."));
    renderPage();
    await screen.findByText(`${RECENT_GENERATIONS.length} questions asked recently`);

    await askQuestion(user);

    await screen.findByText("Couldn't get an answer");
    expect(screen.getByText(`${RECENT_GENERATIONS.length} questions asked recently`)).toBeInTheDocument();
  });

  it("does not retry automatically after a failure", async () => {
    const user = userEvent.setup();
    knowledgeAssistantApi.ask.mockRejectedValue(new Error("Something went wrong."));
    renderPage();

    await askQuestion(user);

    await screen.findByText("Couldn't get an answer");
    expect(knowledgeAssistantApi.ask).toHaveBeenCalledTimes(1);
  });

  // --- Feedback ---

  it('clicking Helpful calls the feedback endpoint with the backend generationId and shows the thanks message', async () => {
    const user = userEvent.setup();
    renderPage();
    await askQuestion(user);
    await screen.findByText("Question");

    await user.click(screen.getByRole("button", { name: "Helpful" }));

    expect(await screen.findByText("Thanks — your feedback was recorded.")).toBeInTheDocument();
    expect(knowledgeAssistantApi.submitFeedback).toHaveBeenCalledWith(
      expect.objectContaining({ generationId: "gen_20260812_abc123", evaluation: "GOOD" }),
    );
    expect(screen.queryByText("Tell us more")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Helpful" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Not helpful" })).toBeDisabled();
  });

  it("clicking Not helpful opens the feedback modal instead of calling the backend immediately", async () => {
    const user = userEvent.setup();
    renderPage();
    await askQuestion(user);
    await screen.findByText("Question");

    await user.click(screen.getByRole("button", { name: "Not helpful" }));

    const modal = screen.getByRole("dialog", { name: "Tell us more" });
    expect(within(modal).getByText("Help us improve this answer.")).toBeInTheDocument();
    expect(within(modal).getByRole("button", { name: "Inaccurate" })).toBeInTheDocument();
    expect(within(modal).getByRole("button", { name: "Missing information" })).toBeInTheDocument();
    expect(within(modal).getByRole("button", { name: "Not relevant" })).toBeInTheDocument();
    expect(within(modal).getByRole("button", { name: "Other" })).toBeInTheDocument();
    expect(within(modal).getByPlaceholderText("What was missing or incorrect?")).toBeInTheDocument();
    expect(screen.queryByText("Thanks — your feedback was recorded.")).not.toBeInTheDocument();
    expect(knowledgeAssistantApi.submitFeedback).not.toHaveBeenCalled();
  });

  it("Submit feedback calls the backend with the generationId, reason, and description, and shows the thanks message only after success", async () => {
    const user = userEvent.setup();
    renderPage();
    await askQuestion(user);
    await screen.findByText("Question");
    await user.click(screen.getByRole("button", { name: "Not helpful" }));

    const modal = screen.getByRole("dialog", { name: "Tell us more" });
    expect(within(modal).getByRole("button", { name: "Submit feedback" })).toBeDisabled();

    await user.click(within(modal).getByRole("button", { name: "Missing information" }));
    await user.type(within(modal).getByPlaceholderText("What was missing or incorrect?"), "Missed the pricing page.");
    expect(within(modal).getByRole("button", { name: "Submit feedback" })).toBeEnabled();

    await user.click(within(modal).getByRole("button", { name: "Submit feedback" }));

    expect(knowledgeAssistantApi.submitFeedback).toHaveBeenCalledWith({
      generationId: "gen_20260812_abc123",
      evaluation: "BAD",
      reason: "MISSING_INFORMATION",
      description: "Missed the pricing page.",
    });
    expect(screen.queryByRole("dialog", { name: "Tell us more" })).not.toBeInTheDocument();
    expect(await screen.findByText("Thanks — your feedback was recorded.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Not helpful" })).toBeDisabled();
  });

  it("Cancel closes the modal without calling the backend", async () => {
    const user = userEvent.setup();
    renderPage();
    await askQuestion(user);
    await screen.findByText("Question");
    await user.click(screen.getByRole("button", { name: "Not helpful" }));
    await user.click(
      within(screen.getByRole("dialog", { name: "Tell us more" })).getByRole("button", { name: "Missing information" }),
    );

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(screen.queryByRole("dialog", { name: "Tell us more" })).not.toBeInTheDocument();
    expect(screen.queryByText("Thanks — your feedback was recorded.")).not.toBeInTheDocument();
    expect(knowledgeAssistantApi.submitFeedback).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Not helpful" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Helpful" })).toBeEnabled();
  });
});
