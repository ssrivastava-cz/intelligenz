import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "./client.js";
import { knowledgeAssistantApi } from "./knowledgeAssistantApi.js";

vi.mock("./client.js", () => ({
  apiClient: {
    get: vi.fn(),
    post: vi.fn(),
  },
}));

describe("knowledgeAssistantApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("ask() posts the question, without a feature or a generationId", async () => {
    apiClient.post.mockResolvedValue({ generationId: "gen_001" });

    await knowledgeAssistantApi.ask({ question: "How does Contact Log work?" });

    expect(apiClient.post).toHaveBeenCalledWith("/knowledge-assistant/ask", {
      userQuestion: "How does Contact Log work?",
      uploadSessionId: undefined,
      topK: undefined,
    });
  });

  it("submitFeedback() posts the generationId returned by ask(), never a question or timestamp", async () => {
    apiClient.post.mockResolvedValue({ generationId: "gen_001", evaluation: "GOOD" });

    await knowledgeAssistantApi.submitFeedback({ generationId: "gen_001", evaluation: "GOOD" });

    expect(apiClient.post).toHaveBeenCalledWith("/knowledge-assistant/feedback", {
      generationId: "gen_001",
      evaluation: "GOOD",
      reason: undefined,
      description: undefined,
    });
  });

  it("submitFeedback() includes reason/description for a BAD evaluation", async () => {
    apiClient.post.mockResolvedValue({ generationId: "gen_001", evaluation: "BAD" });

    await knowledgeAssistantApi.submitFeedback({
      generationId: "gen_001",
      evaluation: "BAD",
      reason: "MISSING_INFORMATION",
      description: "Missed the Bridge functionality.",
    });

    expect(apiClient.post).toHaveBeenCalledWith("/knowledge-assistant/feedback", {
      generationId: "gen_001",
      evaluation: "BAD",
      reason: "MISSING_INFORMATION",
      description: "Missed the Bridge functionality.",
    });
  });

  it("getDebug() requests the debug endpoint for the given generationId", async () => {
    apiClient.get.mockResolvedValue({ generationId: "gen_001" });

    await knowledgeAssistantApi.getDebug("gen_001");

    expect(apiClient.get).toHaveBeenCalledWith("/knowledge-assistant/debug/gen_001");
  });
});
