import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "./client.js";
import { redmineApi } from "./redmineApi.js";

vi.mock("./client.js", () => ({
  apiClient: {
    get: vi.fn(),
  },
}));

describe("redmineApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("getTicketDetail() requests the ticket's debug endpoint", async () => {
    apiClient.get.mockResolvedValue({ ticketId: "12345", description: "Ticket description." });

    const result = await redmineApi.getTicketDetail("12345");

    expect(apiClient.get).toHaveBeenCalledWith("/redmine/12345/debug");
    expect(result.description).toBe("Ticket description.");
  });
});
