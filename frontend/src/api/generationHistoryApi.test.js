import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "./client.js";
import { generationHistoryApi } from "./generationHistoryApi.js";

vi.mock("./client.js", () => ({
  apiClient: {
    get: vi.fn(),
    getBlob: vi.fn(),
  },
}));

describe("generationHistoryApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("list() requests the history endpoint with default pagination/sort params", async () => {
    apiClient.get.mockResolvedValue({ items: [] });

    await generationHistoryApi.list();

    expect(apiClient.get).toHaveBeenCalledWith(
      "/generation/history?page=1&page_size=50&sort=createdAt&order=desc",
    );
  });

  it("list() forwards custom pagination/sort options using snake_case page_size", async () => {
    apiClient.get.mockResolvedValue({ items: [] });

    await generationHistoryApi.list({ page: 2, pageSize: 10, sort: "totalCostUsd", order: "asc" });

    expect(apiClient.get).toHaveBeenCalledWith(
      "/generation/history?page=2&page_size=10&sort=totalCostUsd&order=asc",
    );
  });

  it("stats() requests the stats endpoint", async () => {
    apiClient.get.mockResolvedValue({ totalGenerations: 0 });

    await generationHistoryApi.stats();

    expect(apiClient.get).toHaveBeenCalledWith("/generation/history/stats");
  });

  it("getById() requests the single generation endpoint", async () => {
    apiClient.get.mockResolvedValue({ generationId: "abc" });

    await generationHistoryApi.getById("abc");

    expect(apiClient.get).toHaveBeenCalledWith("/generation/abc");
  });

  it("downloadExcel() fetches the workbook blob and triggers a browser download", async () => {
    const blob = new Blob(["fake-xlsx-bytes"]);
    apiClient.getBlob.mockResolvedValue({
      blob,
      contentDisposition: 'attachment; filename="ADO_TestCases_abc.xlsx"',
    });

    const createObjectURL = vi.fn(() => "blob:mock-url");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { ...URL, createObjectURL, revokeObjectURL });

    const clickSpy = vi.fn();
    const appendSpy = vi.spyOn(document.body, "appendChild").mockImplementation(() => {});
    vi.spyOn(document, "createElement").mockReturnValue({
      set href(_value) {},
      set download(_value) {},
      click: clickSpy,
      remove: vi.fn(),
    });

    await generationHistoryApi.downloadExcel("abc");

    expect(apiClient.getBlob).toHaveBeenCalledWith("/generation/abc/download/excel");
    expect(createObjectURL).toHaveBeenCalledWith(blob);
    expect(clickSpy).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mock-url");

    appendSpy.mockRestore();
    document.createElement.mockRestore();
    vi.unstubAllGlobals();
  });

  it("downloadExcel() falls back to a generated filename when Content-Disposition is missing", async () => {
    apiClient.getBlob.mockResolvedValue({ blob: new Blob(["x"]), contentDisposition: null });

    const createObjectURL = vi.fn(() => "blob:mock-url");
    vi.stubGlobal("URL", { ...URL, createObjectURL, revokeObjectURL: vi.fn() });

    let downloadedFilename;
    vi.spyOn(document, "createElement").mockReturnValue({
      set href(_value) {},
      set download(value) {
        downloadedFilename = value;
      },
      click: vi.fn(),
      remove: vi.fn(),
    });
    vi.spyOn(document.body, "appendChild").mockImplementation(() => {});

    await generationHistoryApi.downloadExcel("xyz");

    expect(downloadedFilename).toBe("ADO_TestCases_xyz.xlsx");

    document.createElement.mockRestore();
    document.body.appendChild.mockRestore();
    vi.unstubAllGlobals();
  });
});
