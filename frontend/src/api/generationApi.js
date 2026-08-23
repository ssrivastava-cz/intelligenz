/**
 * API layer for AI Test Case Generation (`POST /generate`). Generation
 * History's own API module (`generationHistoryApi`) already covers
 * everything needed after a generation exists — fetching its full
 * detail and downloading its Excel workbook — so this module only owns
 * the one call that creates a new generation, plus polling its progress.
 */
import { apiClient, GENERATION_TIMEOUT_MS } from "./client.js";
import { ENDPOINTS } from "./endpoints.js";

export const generationApi = {
  /**
   * Generation involves a real OpenAI Chat completion and can take
   * significantly longer than an ordinary request, so this alone uses
   * the dedicated `GENERATION_TIMEOUT_MS` instead of the default.
   * @param {{feature: string, redmineId?: string, redmineDescription?: string, optionalDescription?: string, uploadSessionId?: string, requestId?: string}} payload
   */
  generate(payload) {
    return apiClient.post(ENDPOINTS.generate, payload, { timeoutMs: GENERATION_TIMEOUT_MS });
  },

  /**
   * Best-effort peek at which internal step an in-flight `generate()`
   * call has reached, for the progress UI to poll — never triggers or
   * duplicates generation itself.
   */
  getProgress(requestId) {
    return apiClient.get(ENDPOINTS.generateProgress(requestId));
  },
};
