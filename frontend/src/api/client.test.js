import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { apiClient, ApiError, GENERATION_TIMEOUT_MS } from "./client.js";
import { generationApi } from "./generationApi.js";

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return {
    ok,
    status,
    statusText: "",
    headers: { get: () => null },
    json: async () => body,
    blob: async () => new Blob(["bytes"]),
  };
}

describe("apiClient", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("get() resolves with the parsed JSON body on success", async () => {
    fetch.mockResolvedValue(jsonResponse({ ok: true }));

    const result = await apiClient.get("/health");

    expect(result).toEqual({ ok: true });
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/health"),
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("post() sends a JSON-encoded body with a Content-Type header", async () => {
    fetch.mockResolvedValue(jsonResponse({ id: "abc" }));

    await apiClient.post("/generate", { feature: "Appointments" });

    const [, options] = fetch.mock.calls[0];
    expect(options.method).toBe("POST");
    expect(options.body).toBe(JSON.stringify({ feature: "Appointments" }));
    expect(options.headers["Content-Type"]).toBe("application/json");
  });

  it("exposes a dedicated 120-second generation timeout, separate from the normal 30-second one", () => {
    expect(GENERATION_TIMEOUT_MS).toBe(120000);
  });

  it("post() uses the default 30-second timeout for ordinary requests", async () => {
    const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");
    fetch.mockResolvedValue(jsonResponse({ id: "abc" }));

    await apiClient.post("/generation/history", { foo: "bar" });

    const [, delay] = setTimeoutSpy.mock.calls[0];
    expect(delay).toBe(30000);
    setTimeoutSpy.mockRestore();
  });

  it("post() uses the dedicated 120-second timeout when timeoutMs is given", async () => {
    const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");
    fetch.mockResolvedValue(jsonResponse({ id: "abc" }));

    await apiClient.post("/generate", { feature: "Appointments" }, { timeoutMs: GENERATION_TIMEOUT_MS });

    const [, delay] = setTimeoutSpy.mock.calls[0];
    expect(delay).toBe(120000);
    setTimeoutSpy.mockRestore();
  });

  it("get() and other non-post calls are unaffected by the generation timeout option", async () => {
    const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");
    fetch.mockResolvedValue(jsonResponse({ ok: true }));

    await apiClient.get("/generation/history/stats");

    const [, delay] = setTimeoutSpy.mock.calls[0];
    expect(delay).toBe(30000);
    setTimeoutSpy.mockRestore();
  });

  it("postForm() sends the given FormData without forcing a Content-Type header", async () => {
    fetch.mockResolvedValue(jsonResponse({ uploadSessionId: "sess-1" }));
    const formData = new FormData();
    formData.append("files", new Blob(["content"]), "notes.md");

    await apiClient.postForm("/uploads", formData);

    const [, options] = fetch.mock.calls[0];
    expect(options.method).toBe("POST");
    expect(options.body).toBe(formData);
    expect(options.headers).toBeUndefined();
  });

  it("getBlob() resolves with the blob and Content-Disposition header", async () => {
    const blob = new Blob(["xlsx-bytes"]);
    fetch.mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: (name) => (name === "Content-Disposition" ? 'attachment; filename="a.xlsx"' : null) },
      blob: async () => blob,
      json: async () => ({}),
    });

    const result = await apiClient.getBlob("/generation/abc/download/excel");

    expect(result.blob).toBe(blob);
    expect(result.contentDisposition).toBe('attachment; filename="a.xlsx"');
  });

  it("throws an ApiError using the server's detail message when the response is not ok", async () => {
    fetch.mockResolvedValue(jsonResponse({ detail: "No Redmine ticket found with id '999'." }, { ok: false, status: 404 }));

    await expect(apiClient.get("/redmine/999/debug")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      message: "No Redmine ticket found with id '999'.",
    });
  });

  it("falls back to a generic status-based message when the error body has no detail", async () => {
    fetch.mockResolvedValue(jsonResponse({}, { ok: false, status: 500 }));

    await expect(apiClient.get("/generate")).rejects.toThrow(/server encountered an error/i);
  });

  it("surfaces FastAPI's own request-validation detail (an array of issues) instead of a generic message", async () => {
    // The exact shape FastAPI produces for a 422 raised by its own
    // request-body validation (before any route handler runs) — a list
    // of {loc, msg, type} issues, not the plain string this app's own
    // errors always use.
    const validationDetail = [
      { loc: ["body", "redmineId"], msg: "Field required", type: "missing" },
      { loc: ["body", "feature"], msg: "Field required", type: "missing" },
    ];
    fetch.mockResolvedValue(jsonResponse({ detail: validationDetail }, { ok: false, status: 422 }));

    await expect(apiClient.post("/generate", { redmineDescription: "x" })).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
      detail: validationDetail,
      message: "redmineId: Field required; feature: Field required",
    });
  });

  it("logs the full failed response (status, headers, body) to the console for diagnosis", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    fetch.mockResolvedValue(
      jsonResponse({ detail: [{ loc: ["body", "feature"], msg: "Field required", type: "missing" }] }, { ok: false, status: 422 }),
    );

    await expect(apiClient.post("/generate", {})).rejects.toThrow();

    expect(consoleErrorSpy).toHaveBeenCalledWith(
      "[apiClient] request failed:",
      expect.objectContaining({
        status: 422,
        body: expect.objectContaining({ detail: expect.any(Array) }),
      }),
    );
    consoleErrorSpy.mockRestore();
  });

  it("falls back to a generic message when the error body is not valid JSON", async () => {
    fetch.mockResolvedValue({
      ok: false,
      status: 502,
      headers: { get: () => null },
      json: async () => {
        throw new Error("not json");
      },
    });

    await expect(apiClient.get("/generate")).rejects.toThrow(ApiError);
  });

  it("wraps a network failure (fetch rejecting) in a friendly ApiError", async () => {
    fetch.mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(apiClient.get("/health")).rejects.toMatchObject({
      name: "ApiError",
      message: expect.stringMatching(/unable to reach the server/i),
    });
  });

  it("wraps an aborted (timed out) request in a friendly ApiError", async () => {
    const abortError = new Error("aborted");
    abortError.name = "AbortError";
    fetch.mockRejectedValue(abortError);

    await expect(apiClient.get("/health")).rejects.toMatchObject({
      name: "ApiError",
      message: expect.stringMatching(/timed out/i),
    });
  });

  it("generationApi.generate() never sends a redmineId/redmineDescription key on the wire when they're undefined", async () => {
    // Regression check for the exact shape the backend requires: an
    // *omitted* field (JSON.stringify drops undefined-valued keys), not
    // an empty string or an explicit null, which is what would actually
    // reach FastAPI's request-body validation for a Feature-only
    // generation (no Redmine Ticket ID entered).
    fetch.mockResolvedValue(jsonResponse({ generationId: "gen-1" }));

    await generationApi.generate({
      feature: "Contact and Sticket Log",
      redmineId: undefined,
      redmineDescription: undefined,
      optionalDescription: undefined,
      uploadSessionId: undefined,
      requestId: "gen-request-abc123",
    });

    const [, options] = fetch.mock.calls[0];
    const sentBody = JSON.parse(options.body);
    expect(sentBody).toEqual({ feature: "Contact and Sticket Log", requestId: "gen-request-abc123" });
    expect(sentBody).not.toHaveProperty("redmineId");
    expect(sentBody).not.toHaveProperty("redmineDescription");
  });
});
