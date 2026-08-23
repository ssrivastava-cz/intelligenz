import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "./client.js";
import { indexingHistoryApi } from "./indexingHistoryApi.js";

vi.mock("./client.js", () => ({
  apiClient: {
    get: vi.fn(),
  },
}));

describe("indexingHistoryApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("list() requests the indexing history summary endpoint", async () => {
    apiClient.get.mockResolvedValue([]);

    await indexingHistoryApi.list();

    expect(apiClient.get).toHaveBeenCalledWith("/index-history/summary");
  });

  it("stats() requests the indexing history stats endpoint", async () => {
    apiClient.get.mockResolvedValue({ totalIndexingRuns: 0 });

    await indexingHistoryApi.stats();

    expect(apiClient.get).toHaveBeenCalledWith("/index-history/stats");
  });
});
