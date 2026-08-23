/**
 * Dedicated API layer for the Knowledge Assistant. All calls go through
 * `apiClient` (the only module allowed to make network requests) — this
 * module just knows which endpoints to hit and how to shape their
 * inputs/outputs, so components never call fetch() directly.
 */
import { apiClient } from "./client.js";
import { ENDPOINTS } from "./endpoints.js";

export const knowledgeAssistantApi = {
  /**
   * Asks a question. The backend is the sole authority for
   * `generationId` — this never sends one, only receives one back.
   * @param {{question: string, feature: string, uploadSessionId?: string, topK?: number}} options
   */
  ask({ question, feature, uploadSessionId, topK } = {}) {
    return apiClient.post(ENDPOINTS.knowledgeAssistantAsk, {
      userQuestion: question,
      feature,
      uploadSessionId: uploadSessionId || undefined,
      topK: topK || undefined,
    });
  },

  /**
   * Records feedback for an already-answered question. `generationId`
   * must be one `ask()` actually returned — never a question, a
   * timestamp, or a frontend-generated id.
   * @param {{generationId: string, evaluation: "GOOD"|"BAD", reason?: string, description?: string}} feedback
   */
  submitFeedback({ generationId, evaluation, reason, description } = {}) {
    return apiClient.post(ENDPOINTS.knowledgeAssistantFeedback, {
      generationId,
      evaluation,
      reason: reason || undefined,
      description: description || undefined,
    });
  },

  /**
   * The real, persisted "Recent Questions and Answers" — newest first,
   * successful generations only (see `GenerationRepository`). Never
   * mock/seeded data; an empty/error result means an empty list, not a
   * fallback to placeholder questions.
   * @param {{limit?: number}} [options]
   */
  listRecent({ limit = 4 } = {}) {
    return apiClient.get(ENDPOINTS.knowledgeAssistantRecent(limit));
  },

  /**
   * Read-only debug detail for one generation — development/debugging
   * use only, never called as part of a normal question submission.
   * @param {string} generationId
   */
  getDebug(generationId) {
    return apiClient.get(ENDPOINTS.knowledgeAssistantDebug(generationId));
  },
};
