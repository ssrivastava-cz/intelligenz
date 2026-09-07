import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { knowledgeAssistantApi } from "../api/knowledgeAssistantApi.js";
import { useKnowledgeAssistant } from "./useKnowledgeAssistant.js";

vi.mock("../api/knowledgeAssistantApi.js", () => ({
  knowledgeAssistantApi: {
    ask: vi.fn(),
    submitFeedback: vi.fn(),
    listRecent: vi.fn(),
  },
}));

function askResponse(overrides = {}) {
  return {
    generationId: "gen_20260812_abc123",
    question: "How is the AI cost calculated for a test plan generation?",
    answer: "Cost is calculated from prompt and completion tokens.",
    sourceDocuments: ["Usage & Billing Guide.pdf"],
    usage: {
      inputTokens: 100,
      outputTokens: 50,
      totalTokens: 150,
      estimatedInputTokens: 120,
      inputCostInr: 0.01,
      outputCostInr: 0.02,
      totalCostInr: 0.03,
    },
    ...overrides,
  };
}

describe("useKnowledgeAssistant", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("starts with an empty conversation", () => {
    const { result } = renderHook(() => useKnowledgeAssistant());

    expect(result.current.messages).toEqual([]);
    expect(result.current.isLoading).toBe(false);
  });

  it("never fetches or seeds from a Recent Questions endpoint on mount", () => {
    renderHook(() => useKnowledgeAssistant());

    expect(knowledgeAssistantApi.listRecent).not.toHaveBeenCalled();
  });

  it("adds the user's message immediately, then an assistant message, without a feature field in the request", async () => {
    knowledgeAssistantApi.ask.mockResolvedValue(askResponse());
    const { result } = renderHook(() => useKnowledgeAssistant());

    act(() => {
      result.current.ask("How is the AI cost calculated for a test plan generation?");
    });

    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0]).toMatchObject({
      role: "user",
      content: "How is the AI cost calculated for a test plan generation?",
    });
    expect(result.current.messages[1]).toMatchObject({ role: "assistant", status: "loading" });
    expect(result.current.isLoading).toBe(true);

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(knowledgeAssistantApi.ask).toHaveBeenCalledWith({
      question: "How is the AI cost calculated for a test plan generation?",
    });
  });

  it("replaces the loading assistant message with the real answer and sources on success", async () => {
    knowledgeAssistantApi.ask.mockResolvedValue(
      askResponse({ sourceDocuments: ["Contact_Log_Workflow.pdf", "Contact_Log_TestCases.xlsx"] }),
    );
    const { result } = renderHook(() => useKnowledgeAssistant());

    await act(async () => {
      await result.current.ask("How does Contact Log work?");
    });

    const [, assistantMessage] = result.current.messages;
    expect(assistantMessage.status).toBe("success");
    expect(assistantMessage.content).toBe("Cost is calculated from prompt and completion tokens.");
    expect(assistantMessage.documents).toEqual([
      { name: "Contact_Log_Workflow.pdf" },
      { name: "Contact_Log_TestCases.xlsx" },
    ]);
    expect(assistantMessage.generationId).toBe("gen_20260812_abc123");
  });

  it("supports multiple user/assistant turns in one conversation", async () => {
    knowledgeAssistantApi.ask
      .mockResolvedValueOnce(askResponse({ answer: "Provider eligibility answer." }))
      .mockResolvedValueOnce(askResponse({ answer: "Inactive provider answer." }));
    const { result } = renderHook(() => useKnowledgeAssistant());

    await act(async () => {
      await result.current.ask("How does provider eligibility work?");
    });
    await act(async () => {
      await result.current.ask("What happens if the provider is inactive?");
    });

    expect(result.current.messages).toHaveLength(4);
    expect(result.current.messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
    expect(result.current.messages[1].content).toBe("Provider eligibility answer.");
    expect(result.current.messages[3].content).toBe("Inactive provider answer.");
  });

  it("ignores a blank question without calling the backend", () => {
    const { result } = renderHook(() => useKnowledgeAssistant());

    act(() => {
      result.current.ask("   ");
    });

    expect(result.current.messages).toEqual([]);
    expect(knowledgeAssistantApi.ask).not.toHaveBeenCalled();
  });

  it("ignores a new question while a request is already in flight", async () => {
    let resolveAsk;
    knowledgeAssistantApi.ask.mockReturnValue(
      new Promise((resolve) => {
        resolveAsk = resolve;
      }),
    );
    const { result } = renderHook(() => useKnowledgeAssistant());

    act(() => {
      result.current.ask("First question?");
    });
    act(() => {
      result.current.ask("Second question while loading?");
    });

    expect(knowledgeAssistantApi.ask).toHaveBeenCalledTimes(1);
    expect(result.current.messages).toHaveLength(2);

    await act(async () => {
      resolveAsk(askResponse());
    });
  });

  it("sets the assistant message to an error state without losing the user's question", async () => {
    knowledgeAssistantApi.ask.mockRejectedValue(new Error("Something went wrong."));
    const { result } = renderHook(() => useKnowledgeAssistant());

    await act(async () => {
      await result.current.ask("How does Contact Log work?");
    });

    expect(result.current.messages[0]).toMatchObject({ role: "user", content: "How does Contact Log work?" });
    expect(result.current.messages[1]).toMatchObject({ role: "assistant", status: "error" });
    expect(result.current.messages[1].errorMessage).toBe("Something went wrong.");
  });

  it("retry re-asks the same question in place, without duplicating the user's message", async () => {
    knowledgeAssistantApi.ask
      .mockRejectedValueOnce(new Error("Something went wrong."))
      .mockResolvedValueOnce(askResponse({ answer: "Recovered answer." }));
    const { result } = renderHook(() => useKnowledgeAssistant());

    await act(async () => {
      await result.current.ask("How does Contact Log work?");
    });
    const failedMessageId = result.current.messages[1].id;

    await act(async () => {
      await result.current.retry(failedMessageId);
    });

    expect(knowledgeAssistantApi.ask).toHaveBeenCalledTimes(2);
    expect(knowledgeAssistantApi.ask).toHaveBeenLastCalledWith({ question: "How does Contact Log work?" });
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1].id).toBe(failedMessageId);
    expect(result.current.messages[1]).toMatchObject({
      role: "assistant",
      status: "success",
      content: "Recovered answer.",
    });
  });

  it("recordFeedback updates the matching assistant message by generationId", async () => {
    knowledgeAssistantApi.ask.mockResolvedValue(askResponse());
    knowledgeAssistantApi.submitFeedback.mockResolvedValue({ generationId: "gen_20260812_abc123", evaluation: "GOOD" });
    const { result } = renderHook(() => useKnowledgeAssistant());

    await act(async () => {
      await result.current.ask("Where can I see the token usage for a run?");
    });

    await act(async () => {
      await result.current.recordFeedback("gen_20260812_abc123", {
        generationId: "gen_20260812_abc123",
        evaluation: "GOOD",
        reason: null,
        description: null,
      });
    });

    expect(knowledgeAssistantApi.submitFeedback).toHaveBeenCalledWith({
      generationId: "gen_20260812_abc123",
      evaluation: "GOOD",
      reason: null,
      description: null,
    });
    expect(result.current.messages[1].feedback).toEqual({
      generationId: "gen_20260812_abc123",
      evaluation: "GOOD",
      reason: null,
      description: null,
    });
  });

  it("recordFeedback leaves feedback unset when the backend call fails", async () => {
    knowledgeAssistantApi.ask.mockResolvedValue(askResponse());
    knowledgeAssistantApi.submitFeedback.mockRejectedValue(new Error("API request failed: 500"));
    const { result } = renderHook(() => useKnowledgeAssistant());

    await act(async () => {
      await result.current.ask("Where can I see the token usage for a run?");
    });

    await act(async () => {
      await result.current.recordFeedback("gen_20260812_abc123", {
        generationId: "gen_20260812_abc123",
        evaluation: "BAD",
        reason: "INACCURATE",
        description: "Wrong.",
      });
    });

    expect(result.current.messages[1].feedback).toBeNull();
  });

  it("startNewConversation clears the conversation back to empty", async () => {
    knowledgeAssistantApi.ask.mockResolvedValue(askResponse());
    const { result } = renderHook(() => useKnowledgeAssistant());

    await act(async () => {
      await result.current.ask("How does Contact Log work?");
    });
    expect(result.current.messages).toHaveLength(2);

    act(() => {
      result.current.startNewConversation();
    });

    expect(result.current.messages).toEqual([]);
  });
});
