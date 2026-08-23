import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { generationApi } from "../api/generationApi.js";
import { generationHistoryApi } from "../api/generationHistoryApi.js";
import { redmineApi } from "../api/redmineApi.js";
import { useTestPlanGeneration } from "./useTestPlanGeneration.js";

vi.mock("../api/generationApi.js", () => ({
  generationApi: { generate: vi.fn(), getProgress: vi.fn() },
}));
vi.mock("../api/generationHistoryApi.js", () => ({
  generationHistoryApi: { getById: vi.fn(), downloadExcel: vi.fn() },
}));
vi.mock("../api/redmineApi.js", () => ({
  redmineApi: { getTicketDetail: vi.fn() },
}));

const TICKET_DETAIL = { ticketId: "12345", description: "Users should be able to reschedule appointments." };
const GENERATE_RESPONSE = { generationId: "gen-1" };
const GENERATION_DETAIL = {
  generationId: "gen-1",
  generationTimeMs: 7200,
  actualUsage: { promptTokens: 1000, completionTokens: 200, totalTokens: 1200, totalCost: { usd: 0.01, inr: 0.87 } },
  estimatedUsage: { estimatedInputCost: { usd: 0.005, inr: 0.44 } },
  testCases: [],
};

describe("useTestPlanGeneration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    generationApi.getProgress.mockResolvedValue({ stage: null });
  });

  it("fetches the primary ticket's description, generates, then loads the full detail", async () => {
    redmineApi.getTicketDetail.mockResolvedValue(TICKET_DETAIL);
    generationApi.generate.mockResolvedValue(GENERATE_RESPONSE);
    generationHistoryApi.getById.mockResolvedValue(GENERATION_DETAIL);
    const { result } = renderHook(() => useTestPlanGeneration());

    act(() => {
      result.current.generate({
        feature: "Appointments",
        ticketId: "12345",
        optionalDescription: "Extra context.",
        uploadSessionId: "sess-1",
      });
    });

    expect(result.current.status).toBe("fetchingTicket");
    await waitFor(() => expect(result.current.status).toBe("success"));

    expect(redmineApi.getTicketDetail).toHaveBeenCalledWith("12345");
    expect(generationApi.generate).toHaveBeenCalledWith(
      expect.objectContaining({
        feature: "Appointments",
        redmineId: "12345",
        redmineDescription: TICKET_DETAIL.description,
        optionalDescription: "Extra context.",
        uploadSessionId: "sess-1",
        requestId: expect.any(String),
      }),
    );
    expect(generationHistoryApi.getById).toHaveBeenCalledWith("gen-1");
    expect(result.current.generationId).toBe("gen-1");
    expect(result.current.detail).toEqual(GENERATION_DETAIL);
  });

  it("generates successfully with no Redmine ticket at all, skipping the ticket lookup entirely", async () => {
    generationApi.generate.mockResolvedValue(GENERATE_RESPONSE);
    generationHistoryApi.getById.mockResolvedValue(GENERATION_DETAIL);
    const { result } = renderHook(() => useTestPlanGeneration());

    act(() => {
      result.current.generate({ feature: "Appointments", ticketId: "", optionalDescription: "" });
    });

    // Skips straight to "generating" — there's no ticket to fetch.
    expect(result.current.status).toBe("generating");
    await waitFor(() => expect(result.current.status).toBe("success"));

    expect(redmineApi.getTicketDetail).not.toHaveBeenCalled();
    expect(generationApi.generate).toHaveBeenCalledWith(
      expect.objectContaining({
        feature: "Appointments",
        redmineId: undefined,
        redmineDescription: undefined,
      }),
    );
    expect(result.current.detail).toEqual(GENERATION_DETAIL);
  });

  it("uses only the first ticket id when multiple comma-separated ids are given", async () => {
    redmineApi.getTicketDetail.mockResolvedValue(TICKET_DETAIL);
    generationApi.generate.mockResolvedValue(GENERATE_RESPONSE);
    generationHistoryApi.getById.mockResolvedValue(GENERATION_DETAIL);
    const { result } = renderHook(() => useTestPlanGeneration());

    await act(async () => {
      await result.current.generate({ feature: "Appointments", ticketId: "12345, 67890", optionalDescription: "" });
    });

    expect(redmineApi.getTicketDetail).toHaveBeenCalledWith("12345");
    expect(generationApi.generate).toHaveBeenCalledWith(
      expect.objectContaining({ redmineId: "12345, 67890" }),
    );
  });

  it("omits optionalDescription/uploadSessionId from the request when not provided", async () => {
    redmineApi.getTicketDetail.mockResolvedValue(TICKET_DETAIL);
    generationApi.generate.mockResolvedValue(GENERATE_RESPONSE);
    generationHistoryApi.getById.mockResolvedValue(GENERATION_DETAIL);
    const { result } = renderHook(() => useTestPlanGeneration());

    await act(async () => {
      await result.current.generate({ feature: "Appointments", ticketId: "12345", optionalDescription: "" });
    });

    expect(generationApi.generate).toHaveBeenCalledWith(
      expect.objectContaining({ optionalDescription: undefined, uploadSessionId: undefined }),
    );
  });

  it("surfaces an error and never calls /generate when the Redmine ticket lookup fails", async () => {
    redmineApi.getTicketDetail.mockRejectedValue(new Error("No Redmine ticket found with id '99999'."));
    const { result } = renderHook(() => useTestPlanGeneration());

    await act(async () => {
      await result.current.generate({ feature: "Appointments", ticketId: "99999", optionalDescription: "" });
    });

    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("No Redmine ticket found with id '99999'.");
    expect(generationApi.generate).not.toHaveBeenCalled();
  });

  it("surfaces an error when generation itself fails", async () => {
    redmineApi.getTicketDetail.mockResolvedValue(TICKET_DETAIL);
    generationApi.generate.mockRejectedValue(new Error("OpenAI chat completion request failed."));
    const { result } = renderHook(() => useTestPlanGeneration());

    await act(async () => {
      await result.current.generate({ feature: "Appointments", ticketId: "12345", optionalDescription: "" });
    });

    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("OpenAI chat completion request failed.");
  });

  it("downloadExcel() calls generationHistoryApi with the current generation id", async () => {
    redmineApi.getTicketDetail.mockResolvedValue(TICKET_DETAIL);
    generationApi.generate.mockResolvedValue(GENERATE_RESPONSE);
    generationHistoryApi.getById.mockResolvedValue(GENERATION_DETAIL);
    generationHistoryApi.downloadExcel.mockResolvedValue(undefined);
    const { result } = renderHook(() => useTestPlanGeneration());
    await act(async () => {
      await result.current.generate({ feature: "Appointments", ticketId: "12345", optionalDescription: "" });
    });

    await act(async () => {
      await result.current.downloadExcel();
    });

    expect(generationHistoryApi.downloadExcel).toHaveBeenCalledWith("gen-1");
    expect(result.current.downloadStatus).toBe("idle");
  });

  it("downloadExcel() surfaces a friendly error on failure", async () => {
    redmineApi.getTicketDetail.mockResolvedValue(TICKET_DETAIL);
    generationApi.generate.mockResolvedValue(GENERATE_RESPONSE);
    generationHistoryApi.getById.mockResolvedValue(GENERATION_DETAIL);
    generationHistoryApi.downloadExcel.mockRejectedValue(new Error("The requested resource could not be found."));
    const { result } = renderHook(() => useTestPlanGeneration());
    await act(async () => {
      await result.current.generate({ feature: "Appointments", ticketId: "12345", optionalDescription: "" });
    });

    await act(async () => {
      await result.current.downloadExcel();
    });

    expect(result.current.downloadStatus).toBe("error");
    expect(result.current.downloadError).toBe("The requested resource could not be found.");
  });

  describe("progress polling", () => {
    afterEach(() => {
      vi.useRealTimers();
    });

    it("polls generationApi.getProgress while the request is in flight, tracking the real backend stage", async () => {
      vi.useFakeTimers();
      redmineApi.getTicketDetail.mockResolvedValue(TICKET_DETAIL);
      let resolveGenerate;
      generationApi.generate.mockReturnValue(
        new Promise((resolve) => {
          resolveGenerate = resolve;
        }),
      );
      generationApi.getProgress.mockResolvedValue({ stage: "generatingTestCases" });
      generationHistoryApi.getById.mockResolvedValue(GENERATION_DETAIL);

      const { result } = renderHook(() => useTestPlanGeneration());

      act(() => {
        result.current.generate({ feature: "Appointments", ticketId: "12345", optionalDescription: "" });
      });

      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(result.current.status).toBe("generating");

      await act(async () => {
        await vi.advanceTimersByTimeAsync(400);
      });
      expect(generationApi.getProgress).toHaveBeenCalled();
      expect(result.current.generationStage).toBe("generatingTestCases");

      const pollCountBeforeSettling = generationApi.getProgress.mock.calls.length;

      await act(async () => {
        resolveGenerate(GENERATE_RESPONSE);
        await vi.advanceTimersByTimeAsync(0);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });

      // Never polls again once the request has settled — this is a
      // status-only poll, never a mechanism that could re-trigger
      // generation.
      expect(generationApi.getProgress.mock.calls.length).toBe(pollCountBeforeSettling);
      expect(result.current.status).toBe("success");
    });

    it("moves to loadingResult (stage: complete) once POST /generate resolves, before results are shown", async () => {
      vi.useFakeTimers();
      redmineApi.getTicketDetail.mockResolvedValue(TICKET_DETAIL);
      generationApi.generate.mockResolvedValue(GENERATE_RESPONSE);
      generationApi.getProgress.mockResolvedValue({ stage: "savingHistory" });
      let resolveDetail;
      generationHistoryApi.getById.mockReturnValue(
        new Promise((resolve) => {
          resolveDetail = resolve;
        }),
      );

      const { result } = renderHook(() => useTestPlanGeneration());

      act(() => {
        result.current.generate({ feature: "Appointments", ticketId: "12345", optionalDescription: "" });
      });

      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });

      expect(result.current.status).toBe("loadingResult");
      expect(result.current.generationStage).toBe("complete");

      await act(async () => {
        resolveDetail(GENERATION_DETAIL);
        await vi.advanceTimersByTimeAsync(0);
      });

      expect(result.current.status).toBe("success");
    });
  });
});
