import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "./client.js";
import { usageApi } from "./usageApi.js";

vi.mock("./client.js", () => ({
  apiClient: {
    get: vi.fn(),
  },
}));

describe("usageApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("get() requests the unified usage endpoint", async () => {
    apiClient.get.mockResolvedValue({ summary: {}, generations: [] });

    await usageApi.get();

    expect(apiClient.get).toHaveBeenCalledWith("/usage");
  });
});
