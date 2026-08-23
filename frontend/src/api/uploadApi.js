/**
 * API layer for the Uploaded Document pipeline: upload files, preview
 * their embedding cost, then (only once the user confirms) actually
 * embed them. All calls go through `apiClient` — components never call
 * fetch() directly.
 */
import { apiClient } from "./client.js";
import { ENDPOINTS } from "./endpoints.js";

export const uploadApi = {
  /**
   * Uploads one or more files and creates a new upload session for
   * them. Never embeds — that's a separate, explicit step.
   * @param {File[]} files
   */
  uploadFiles(files) {
    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }
    return apiClient.postForm(ENDPOINTS.uploads, formData);
  },

  /**
   * Previews token usage and estimated cost for an upload session's
   * documents. Local token counting only — no OpenAI call, no embedding.
   */
  getEmbeddingPreview(uploadSessionId) {
    return apiClient.post(ENDPOINTS.uploadEmbeddingPreview(uploadSessionId));
  },

  /**
   * Runs the real embedding pipeline for an upload session — the only
   * upload-related call that costs money. Only ever invoked after the
   * user explicitly confirms, never automatically.
   */
  embedDocuments(uploadSessionId) {
    return apiClient.post(ENDPOINTS.uploadEmbed(uploadSessionId));
  },
};
