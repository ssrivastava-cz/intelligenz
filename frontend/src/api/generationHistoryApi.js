/**
 * Dedicated API layer for the Generation History page. All calls go
 * through `apiClient` (the only module allowed to make network
 * requests) — this module just knows which endpoints to hit and how to
 * shape their inputs/outputs, so components never call fetch() directly.
 */
import { apiClient } from "./client.js";
import { ENDPOINTS } from "./endpoints.js";

const DEFAULT_PAGE_SIZE = 50;

function extractFilename(contentDisposition, fallback) {
  const match = contentDisposition?.match(/filename="?([^"]+)"?/);
  return match?.[1] ?? fallback;
}

function triggerBlobDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export const generationHistoryApi = {
  /**
   * Paginated, sortable list of generation summaries.
   * @param {{page?: number, pageSize?: number, sort?: string, order?: "asc"|"desc"}} options
   */
  list({ page = 1, pageSize = DEFAULT_PAGE_SIZE, sort = "createdAt", order = "desc" } = {}) {
    const params = new URLSearchParams({ page, page_size: pageSize, sort, order });
    return apiClient.get(`${ENDPOINTS.generationHistory}?${params.toString()}`);
  },

  /** Dashboard aggregates (total generations, spend, averages, ...). */
  stats() {
    return apiClient.get(ENDPOINTS.generationHistoryStats);
  },

  /** The complete persisted generation — prompt/response are present but unused by this UI. */
  getById(generationId) {
    return apiClient.get(ENDPOINTS.generationById(generationId));
  },

  /**
   * Downloads the persisted ADO Excel workbook for one generation.
   * Never regenerates anything — purely fetches and saves the file
   * already produced by a prior `POST /generate`.
   */
  async downloadExcel(generationId) {
    const { blob, contentDisposition } = await apiClient.getBlob(ENDPOINTS.generationExcelDownload(generationId));
    triggerBlobDownload(blob, extractFilename(contentDisposition, `ADO_TestCases_${generationId}.xlsx`));
  },
};
