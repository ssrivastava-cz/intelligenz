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

const RECENT_GENERATIONS = [
  {
    generationId: "gen_20260810_real1",
    question: "How is the AI cost calculated for a test plan generation?",
    answer: "Cost is calculated from prompt and completion tokens.",
    sourceDocuments: ["Usage & Billing Guide.pdf"],
    createdAt: "2026-08-10T21:20:00Z",
  },
  {
    generationId: "gen_20260809_real2",
    question: "Where can I see the token usage for a specific test plan run?",
    answer: "Open the generation's entry in Test Plan Generation History.",
    sourceDocuments: [],
    createdAt: "2026-08-09T09:00:00Z",
  },
];

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
    knowledgeAssistantApi.listRecent.mockResolvedValue(RECENT_GENERATIONS);
  });

  it("starts idle with no current answer", () => {
    const { result } = renderHook(() => useKnowledgeAssistant());

    expect(result.current.status).toBe("idle");
    expect(result.current.currentAnswer).toBeNull();
  });

  it("fetches the real recent generations from the backend on mount, never mock data", async () => {
    const { result } = renderHook(() => useKnowledgeAssistant());

    await waitFor(() => expect(result.current.recentQuestions).toHaveLength(RECENT_GENERATIONS.length));

    expect(knowledgeAssistantApi.listRecent).toHaveBeenCalled();
    expect(result.current.recentQuestions[0]).toMatchObject({
      generationId: "gen_20260810_real1",
      question: "How is the AI cost calculated for a test plan generation?",
      answer: "Cost is calculated from prompt and completion tokens.",
      createdAt: "2026-08-10T21:20:00Z",
    });
    expect(result.current.recentQuestions[0].documents).toEqual([{ name: "Usage & Billing Guide.pdf" }]);
  });

  it("leaves the recent list empty, never falling back to mock data, when the backend fetch fails", async () => {
    knowledgeAssistantApi.listRecent.mockRejectedValue(new Error("Something went wrong."));
    const { result } = renderHook(() => useKnowledgeAssistant());

    await waitFor(() => expect(knowledgeAssistantApi.listRecent).toHaveBeenCalled());

    expect(result.current.recentQuestions).toEqual([]);
  });

  it("asking a question calls the real backend (not the mock) and sets the current answer", async () => {
    knowledgeAssistantApi.ask.mockResolvedValue(
      askResponse({ question: "How is the AI cost calculated for a test plan generation?" }),
    );
    const { result } = renderHook(() => useKnowledgeAssistant());
    await waitFor(() => expect(result.current.recentQuestions).toHaveLength(RECENT_GENERATIONS.length));

    act(() => {
      result.current.ask("How is the AI cost calculated for a test plan generation?", "Contact Log");
    });
    expect(result.current.status).toBe("loading");

    await waitFor(() => expect(result.current.status).toBe("success"));

    expect(knowledgeAssistantApi.ask).toHaveBeenCalledWith({
      question: "How is the AI cost calculated for a test plan generation?",
      feature: "Contact Log",
    });
    expect(result.current.currentAnswer.question).toBe("How is the AI cost calculated for a test plan generation?");
    expect(result.current.currentAnswer.generationId).toBe("gen_20260812_abc123");
    expect(result.current.recentQuestions[0].generationId).toBe(result.current.currentAnswer.generationId);
    expect(result.current.recentQuestions).toHaveLength(RECENT_GENERATIONS.length + 1);
  });

  it("maps the backend's sourceDocuments into the document list the UI expects", async () => {
    knowledgeAssistantApi.ask.mockResolvedValue(
      askResponse({ sourceDocuments: ["Contact_Log_Workflow.pdf", "Contact_Log_TestCases.xlsx"] }),
    );
    const { result } = renderHook(() => useKnowledgeAssistant());

    await act(async () => {
      await result.current.ask("How does Contact Log work?", "Contact Log");
    });

    expect(result.current.currentAnswer.documents).toEqual([
      { name: "Contact_Log_Workflow.pdf" },
      { name: "Contact_Log_TestCases.xlsx" },
    ]);
  });

  it("ignores a blank question without calling the backend", async () => {
    const { result } = renderHook(() => useKnowledgeAssistant());

    act(() => {
      result.current.ask("   ", "Contact Log");
    });

    expect(result.current.status).toBe("idle");
    expect(result.current.currentAnswer).toBeNull();
    expect(knowledgeAssistantApi.ask).not.toHaveBeenCalled();
  });

  it("ignores a question with no feature selected, without calling the backend", async () => {
    const { result } = renderHook(() => useKnowledgeAssistant());

    act(() => {
      result.current.ask("How does Contact Log work?", "");
    });

    expect(result.current.status).toBe("idle");
    expect(knowledgeAssistantApi.ask).not.toHaveBeenCalled();
  });

  it("sets an error status and message when the backend call fails, without a fake answer", async () => {
    knowledgeAssistantApi.ask.mockRejectedValue(new Error("Something went wrong."));
    const { result } = renderHook(() => useKnowledgeAssistant());
    await waitFor(() => expect(result.current.recentQuestions).toHaveLength(RECENT_GENERATIONS.length));

    act(() => {
      result.current.ask("How does Contact Log work?", "Contact Log");
    });
    await waitFor(() => expect(result.current.status).toBe("error"));

    expect(result.current.error).toBe("Something went wrong.");
    expect(result.current.currentAnswer).toBeNull();
    // The failed question was not added to the recent list.
    expect(result.current.recentQuestions).toHaveLength(RECENT_GENERATIONS.length);
  });

  it("recordFeedback calls the backend with the generationId and only updates state on success", async () => {
    knowledgeAssistantApi.ask.mockResolvedValue(askResponse());
    knowledgeAssistantApi.submitFeedback.mockResolvedValue({ generationId: "gen_20260812_abc123", evaluation: "GOOD" });
    const { result } = renderHook(() => useKnowledgeAssistant());

    await act(async () => {
      await result.current.ask("Where can I see the token usage for a run?", "Contact Log");
    });
    const { generationId } = result.current.currentAnswer;

    await act(async () => {
      await result.current.recordFeedback(generationId, {
        generationId,
        evaluation: "GOOD",
        reason: null,
        description: null,
      });
    });

    expect(knowledgeAssistantApi.submitFeedback).toHaveBeenCalledWith({
      generationId,
      evaluation: "GOOD",
      reason: null,
      description: null,
    });
    expect(result.current.currentAnswer.feedback).toEqual({
      generationId,
      evaluation: "GOOD",
      reason: null,
      description: null,
    });
    // Same record everywhere — the recent list entry for this
    // generationId reflects the feedback too, not a stale duplicate.
    const recentEntry = result.current.recentQuestions.find((item) => item.generationId === generationId);
    expect(recentEntry.feedback.evaluation).toBe("GOOD");
  });

  it("recordFeedback leaves feedback unset when the backend call fails", async () => {
    knowledgeAssistantApi.ask.mockResolvedValue(askResponse());
    knowledgeAssistantApi.submitFeedback.mockRejectedValue(new Error("API request failed: 500"));
    const { result } = renderHook(() => useKnowledgeAssistant());

    await act(async () => {
      await result.current.ask("Where can I see the token usage for a run?", "Contact Log");
    });
    const { generationId } = result.current.currentAnswer;

    await act(async () => {
      await result.current.recordFeedback(generationId, {
        generationId,
        evaluation: "BAD",
        reason: "INACCURATE",
        description: "Wrong.",
      });
    });

    expect(result.current.currentAnswer.feedback).toBeNull();
  });

  it("caps the recent questions list rather than growing it unbounded", async () => {
    knowledgeAssistantApi.ask.mockImplementation(({ question }) => Promise.resolve(askResponse({ question })));
    const { result } = renderHook(() => useKnowledgeAssistant());
    await waitFor(() => expect(result.current.recentQuestions).toHaveLength(RECENT_GENERATIONS.length));
    const initialLength = result.current.recentQuestions.length;

    for (let i = 0; i < initialLength + 8; i += 1) {
      await act(async () => {
        await result.current.ask(`Question number ${i}`, "Contact Log");
      });
    }

    expect(result.current.recentQuestions.length).toBeLessThanOrEqual(8);
  });
});
