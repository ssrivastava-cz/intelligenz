/**
 * Dedicated API layer for Document Indexing & Embedding history — a
 * separate AI activity type from Test Plan Generation (see
 * `generationHistoryApi.js`). All calls go through `apiClient`.
 */
import { apiClient } from "./client.js";
import { ENDPOINTS } from "./endpoints.js";

export const indexingHistoryApi = {
  /** Every persisted, successful indexing run, newest first, with its INR cost. */
  list() {
    return apiClient.get(ENDPOINTS.indexingHistory);
  },

  /** Dashboard aggregates (total runs, documents, chunks, tokens, embedding cost). */
  stats() {
    return apiClient.get(ENDPOINTS.indexingHistoryStats);
  },
};
