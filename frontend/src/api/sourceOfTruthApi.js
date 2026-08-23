/**
 * API layer for the Source of Truth knowledge base — currently only
 * used to power the Test Plan Generator's Feature dropdown with real
 * indexed features instead of a hardcoded list.
 */
import { apiClient } from "./client.js";
import { ENDPOINTS } from "./endpoints.js";

export const sourceOfTruthApi = {
  /** Every feature the knowledge base currently has documents for. */
  listFeatures() {
    return apiClient.get(ENDPOINTS.sourceOfTruthFeatures);
  },
};
