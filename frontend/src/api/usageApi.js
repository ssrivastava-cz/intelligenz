/**
 * Dedicated API layer for the unified Usage Dashboard view — combines
 * Test Plan Generator and Knowledge Assistant usage (see backend's
 * `UsageService`) into one response. All calls go through `apiClient`.
 */
import { apiClient } from "./client.js";
import { ENDPOINTS } from "./endpoints.js";

export const usageApi = {
  /** `{ summary, generations }` — combined usage across both sources, newest first. */
  get() {
    return apiClient.get(ENDPOINTS.usage);
  },
};
