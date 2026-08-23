import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "./client.js";
import { sourceOfTruthApi } from "./sourceOfTruthApi.js";

vi.mock("./client.js", () => ({
  apiClient: {
    get: vi.fn(),
  },
}));

describe("sourceOfTruthApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("listFeatures() requests the source-of-truth feature list", async () => {
    apiClient.get.mockResolvedValue([{ feature: "Appointments", documents: 3 }]);

    const result = await sourceOfTruthApi.listFeatures();

    expect(apiClient.get).toHaveBeenCalledWith("/source-of-truth");
    expect(result).toEqual([{ feature: "Appointments", documents: 3 }]);
  });
});
