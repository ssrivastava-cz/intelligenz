/**
 * Central registry of backend API endpoint paths.
 * Keep endpoint strings here instead of scattering them across components.
 */
export const ENDPOINTS = {
  health: "/health",

  // Uploads
  uploads: "/uploads",
  uploadEmbeddingPreview: (uploadSessionId) => `/uploads/${uploadSessionId}/embedding-preview`,
  uploadEmbed: (uploadSessionId) => `/uploads/${uploadSessionId}/embed`,

  // AI Generation
  generate: "/generate",
  generateProgress: (requestId) => `/generate/progress/${requestId}`,

  // Generation History
  generationHistory: "/generation/history",
  generationHistoryStats: "/generation/history/stats",
  generationById: (generationId) => `/generation/${generationId}`,
  generationExcelDownload: (generationId) => `/generation/${generationId}/download/excel`,

  // Usage Dashboard — unified view combining Test Plan Generator and
  // Knowledge Assistant usage into one response (see `usageApi.js`).
  usage: "/usage",

  // Document Indexing / Embedding history (Usage Dashboard) — a
  // separate AI activity type from generation; never a generation's cost.
  indexingHistory: "/index-history/summary",
  indexingHistoryStats: "/index-history/stats",

  // Source of Truth (feature list for the Generator's Feature dropdown)
  sourceOfTruthFeatures: "/source-of-truth",

  // Redmine (ticket description lookup, so the generator doesn't require
  // the user to retype what's already in the ticket)
  redmineTicketDebug: (ticketId) => `/redmine/${ticketId}/debug`,

  // Knowledge Assistant — the backend is the sole authority for
  // generationId (returned by `ask`, never sent to it); `feedback` and
  // `debug` are both keyed by whatever `ask` returned.
  knowledgeAssistantAsk: "/knowledge-assistant/ask",
  knowledgeAssistantFeedback: "/knowledge-assistant/feedback",
  knowledgeAssistantRecent: (limit) => `/knowledge-assistant/recent?limit=${limit}`,
  knowledgeAssistantDebug: (generationId) => `/knowledge-assistant/debug/${generationId}`,
};
