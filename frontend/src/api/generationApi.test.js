import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiClient, GENERATION_TIMEOUT_MS } from "./client.js";
import { generationApi } from "./generationApi.js";

vi.mock("./client.js", async () => {
  const actual = await vi.importActual("./client.js");
  return {
    ...actual,
    apiClient: {
      post: vi.fn(),
      get: vi.fn(),
    },
  };
});

describe("generationApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("generate() posts the payload to /generate using the dedicated generation timeout", async () => {
    apiClient.post.mockResolvedValue({ generationId: "gen-1" });
    const payload = {
      feature: "Appointments",
      redmineId: "12345",
      redmineDescription: "Ticket description.",
      optionalDescription: "Extra context.",
      uploadSessionId: "sess-1",
    };

    const result = await generationApi.generate(payload);

    expect(apiClient.post).toHaveBeenCalledWith("/generate", payload, { timeoutMs: GENERATION_TIMEOUT_MS });
    expect(result).toEqual({ generationId: "gen-1" });
  });

  it("generate() works without a Redmine ticket, since it's now optional", async () => {
    apiClient.post.mockResolvedValue({ generationId: "gen-2" });
    const payload = { feature: "Appointments" };

    await generationApi.generate(payload);

    expect(apiClient.post).toHaveBeenCalledWith("/generate", payload, { timeoutMs: GENERATION_TIMEOUT_MS });
  });

  it("getProgress() requests the progress endpoint for the given request id", async () => {
    apiClient.get.mockResolvedValue({ stage: "generatingTestCases" });

    const result = await generationApi.getProgress("req-1");

    expect(apiClient.get).toHaveBeenCalledWith("/generate/progress/req-1");
    expect(result).toEqual({ stage: "generatingTestCases" });
  });
});
