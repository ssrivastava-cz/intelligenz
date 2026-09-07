import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { knowledgeAssistantApi } from "../../api/knowledgeAssistantApi.js";
import KnowledgeAssistant from "./KnowledgeAssistant.jsx";

vi.mock("../../api/knowledgeAssistantApi.js", () => ({
  knowledgeAssistantApi: {
    ask: vi.fn(),
    submitFeedback: vi.fn(),
    getDebug: vi.fn(),
    listRecent: vi.fn(),
  },
}));

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

function getComposerInput() {
  return screen.getByRole("textbox");
}

async function askQuestion(user, question = "How does Contact Log handle bridged contacts?") {
  await user.type(getComposerInput(), question);
  await user.click(screen.getByRole("button", { name: "Send" }));
}

describe("KnowledgeAssistant page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    knowledgeAssistantApi.ask.mockResolvedValue(ASK_RESPONSE);
    knowledgeAssistantApi.submitFeedback.mockResolvedValue({
      generationId: ASK_RESPONSE.generationId,
      evaluation: "GOOD",
    });
  });

  // --- Empty state ---

  it("renders a clean empty state with example prompts and no feature dropdown when there are no messages", () => {
    renderPage();

    expect(screen.getByText("How can I help you?")).toBeInTheDocument();
    expect(screen.getByText("Release Team Intelligenz")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "How does provider eligibility work?" })).toBeInTheDocument();
    expect(screen.queryByLabelText(/feature/i)).not.toBeInTheDocument();
    expect(screen.queryByText("Feature")).not.toBeInTheDocument();
  });

  it("never fetches a Recent Questions and Answers list", () => {
    renderPage();

    expect(knowledgeAssistantApi.listRecent).not.toHaveBeenCalled();
    expect(screen.queryByText(/questions asked recently/)).not.toBeInTheDocument();
  });

  it("clicking an example prompt asks it immediately", async () => {
    const user = userEvent.setup();
    renderPage();

    await user.click(screen.getByRole("button", { name: "How does provider eligibility work?" }));

    expect(knowledgeAssistantApi.ask).toHaveBeenCalledWith({ question: "How does provider eligibility work?" });
    expect(await screen.findByText("How does provider eligibility work?")).toBeInTheDocument();
  });

  // --- Sending a question ---

  it("sends the question to the real backend without a feature field", async () => {
    const user = userEvent.setup();
    renderPage();

    await askQuestion(user);

    expect(knowledgeAssistantApi.ask).toHaveBeenCalledWith({
      question: "How does Contact Log handle bridged contacts?",
    });
  });

  it("shows the user's message immediately and clears the composer", async () => {
    const user = userEvent.setup();
    renderPage();

    await askQuestion(user);

    expect(screen.getByText("How does Contact Log handle bridged contacts?")).toBeInTheDocument();
    expect(getComposerInput()).toHaveValue("");
  });

  it("shows a loading indicator while waiting for the API, then the real answer", async () => {
    let resolveAsk;
    knowledgeAssistantApi.ask.mockReturnValue(
      new Promise((resolve) => {
        resolveAsk = resolve;
      }),
    );
    const user = userEvent.setup();
    renderPage();

    await askQuestion(user);

    expect(screen.getByText("Searching the knowledge base…")).toBeInTheDocument();

    await user.type(getComposerInput(), "irrelevant while loading");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(knowledgeAssistantApi.ask).toHaveBeenCalledTimes(1);

    resolveAsk(ASK_RESPONSE);
    await screen.findByText("According to the workflow, bridged contacts are merged automatically.");
    expect(screen.queryByText("Searching the knowledge base…")).not.toBeInTheDocument();
  });

  it("renders every source document the backend returned", async () => {
    const user = userEvent.setup();
    renderPage();

    await askQuestion(user);

    await screen.findByText("Sources");
    expect(screen.getByText("Contact_Log_Workflow.pdf")).toBeInTheDocument();
    expect(screen.getByText("Contact_Log_TestCases.xlsx")).toBeInTheDocument();
  });

  it("supports multiple follow-up questions in one conversation", async () => {
    const user = userEvent.setup();
    knowledgeAssistantApi.ask
      .mockResolvedValueOnce({ ...ASK_RESPONSE, answer: "First answer." })
      .mockResolvedValueOnce({ ...ASK_RESPONSE, answer: "Second answer." });
    renderPage();

    await askQuestion(user, "How does provider eligibility work?");
    await screen.findByText("First answer.");

    await askQuestion(user, "What about inactive providers?");
    await screen.findByText("Second answer.");

    expect(screen.getByText("How does provider eligibility work?")).toBeInTheDocument();
    expect(screen.getByText("What about inactive providers?")).toBeInTheDocument();
    expect(screen.getByText("First answer.")).toBeInTheDocument();
    expect(screen.getByText("Second answer.")).toBeInTheDocument();
  });

  // --- Composer keyboard behavior ---

  it("Enter submits the message", async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(getComposerInput(), "How does Contact Log work?{Enter}");

    expect(knowledgeAssistantApi.ask).toHaveBeenCalledWith({ question: "How does Contact Log work?" });
  });

  it("Shift+Enter inserts a newline instead of submitting", async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(getComposerInput(), "First line{Shift>}{Enter}{/Shift}Second line");

    expect(knowledgeAssistantApi.ask).not.toHaveBeenCalled();
    expect(getComposerInput()).toHaveValue("First line\nSecond line");
  });

  it("does not submit an empty or whitespace-only message", async () => {
    const user = userEvent.setup();
    renderPage();

    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();

    await user.type(getComposerInput(), "   ");
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(knowledgeAssistantApi.ask).not.toHaveBeenCalled();
  });

  it("disables Send while a request is in flight, and re-enables it afterward", async () => {
    let resolveAsk;
    knowledgeAssistantApi.ask.mockReturnValue(
      new Promise((resolve) => {
        resolveAsk = resolve;
      }),
    );
    const user = userEvent.setup();
    renderPage();

    await askQuestion(user);

    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();

    resolveAsk(ASK_RESPONSE);
    await screen.findByText("According to the workflow, bridged contacts are merged automatically.");
    await user.type(getComposerInput(), "another question");
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();
  });

  // --- Errors ---

  it("keeps the user's message and shows a clear error state when the API fails", async () => {
    const user = userEvent.setup();
    knowledgeAssistantApi.ask.mockRejectedValue(new Error("Something went wrong."));
    renderPage();

    await askQuestion(user);

    expect(await screen.findByText("Something went wrong.")).toBeInTheDocument();
    expect(screen.getByText("How does Contact Log handle bridged contacts?")).toBeInTheDocument();
  });

  it("Retry re-asks the failed question and can succeed", async () => {
    const user = userEvent.setup();
    knowledgeAssistantApi.ask.mockRejectedValueOnce(new Error("Something went wrong."));
    knowledgeAssistantApi.ask.mockResolvedValueOnce(ASK_RESPONSE);
    renderPage();

    await askQuestion(user);
    await screen.findByText("Something went wrong.");

    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(knowledgeAssistantApi.ask).toHaveBeenCalledTimes(2);
    expect(await screen.findByText("According to the workflow, bridged contacts are merged automatically.")).toBeInTheDocument();
    // Only one copy of the user's question — retry didn't duplicate it.
    expect(screen.getAllByText("How does Contact Log handle bridged contacts?")).toHaveLength(1);
  });

  // --- New chat ---

  it("New chat clears the current conversation back to the empty state", async () => {
    const user = userEvent.setup();
    renderPage();

    await askQuestion(user);
    await screen.findByText("According to the workflow, bridged contacts are merged automatically.");

    await user.click(screen.getByRole("button", { name: /new chat/i }));

    expect(screen.queryByText("How does Contact Log handle bridged contacts?")).not.toBeInTheDocument();
    expect(screen.getByText("How can I help you?")).toBeInTheDocument();
  });

  // --- Feedback ---

  it("clicking Helpful calls the feedback endpoint with the backend generationId", async () => {
    const user = userEvent.setup();
    renderPage();
    await askQuestion(user);
    await screen.findByText("According to the workflow, bridged contacts are merged automatically.");

    await user.click(screen.getByRole("button", { name: "Helpful" }));

    expect(await screen.findByText("Thanks — your feedback was recorded.")).toBeInTheDocument();
    expect(knowledgeAssistantApi.submitFeedback).toHaveBeenCalledWith(
      expect.objectContaining({ generationId: "gen_20260812_abc123", evaluation: "GOOD" }),
    );
  });

  it("clicking Not helpful opens the feedback modal instead of calling the backend immediately", async () => {
    const user = userEvent.setup();
    renderPage();
    await askQuestion(user);
    await screen.findByText("According to the workflow, bridged contacts are merged automatically.");

    await user.click(screen.getByRole("button", { name: "Not helpful" }));

    const modal = screen.getByRole("dialog", { name: "Tell us more" });
    expect(within(modal).getByRole("button", { name: "Missing information" })).toBeInTheDocument();
    expect(knowledgeAssistantApi.submitFeedback).not.toHaveBeenCalled();
  });

  it("Submit feedback calls the backend with the generationId, reason, and description", async () => {
    const user = userEvent.setup();
    renderPage();
    await askQuestion(user);
    await screen.findByText("According to the workflow, bridged contacts are merged automatically.");
    await user.click(screen.getByRole("button", { name: "Not helpful" }));

    const modal = screen.getByRole("dialog", { name: "Tell us more" });
    await user.click(within(modal).getByRole("button", { name: "Missing information" }));
    await user.type(within(modal).getByPlaceholderText("What was missing or incorrect?"), "Missed the pricing page.");
    await user.click(within(modal).getByRole("button", { name: "Submit feedback" }));

    expect(knowledgeAssistantApi.submitFeedback).toHaveBeenCalledWith({
      generationId: "gen_20260812_abc123",
      evaluation: "BAD",
      reason: "MISSING_INFORMATION",
      description: "Missed the pricing page.",
    });
    expect(await screen.findByText("Thanks — your feedback was recorded.")).toBeInTheDocument();
  });
});
