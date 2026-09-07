/**
 * Single HTTP client for talking to the FastAPI backend.
 * This is the ONLY module allowed to make network requests — the
 * frontend must never call OpenAI or MongoDB directly.
 */
// Origin of the FastAPI backend — e.g. `http://localhost:8000` in local dev,
// or the deployed backend URL in production (set `VITE_API_URL` in the Netlify
// site environment variables). `VITE_*` vars are inlined into the built bundle
// at build time and are therefore PUBLIC — never put a secret in one.
// All backend routes live under the `/api/v1` prefix, which the frontend
// appends here rather than baking into the env value.
const API_ORIGIN = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");
const API_BASE_URL = `${API_ORIGIN}/api/v1`;
const REQUEST_TIMEOUT_MS = 30000;
// AI Test Case Generation takes significantly longer than an ordinary
// API call (a real OpenAI Chat completion), so it alone opts into this
// longer timeout via `apiClient.post(path, body, { timeoutMs: GENERATION_TIMEOUT_MS })`
// — every other request keeps the normal `REQUEST_TIMEOUT_MS` above.
export const GENERATION_TIMEOUT_MS = 120000;

/**
 * Thrown for every failed request — network failure, timeout, or a
 * non-2xx response — so every caller can rely on one error shape
 * instead of each API module parsing responses itself. `message` is
 * always a safe, readable string to show a user directly; `status` and
 * `detail` are there for callers that need the raw failure (`detail` is
 * whatever the backend's `detail` field actually was — a plain string
 * for this app's own errors, or FastAPI's own array of `{loc, msg,
 * type}` issues for a request-body validation failure).
 */
export class ApiError extends Error {
  constructor(message, { status = null, detail = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function readErrorBody(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * FastAPI's own request-body validation errors (raised before a route
 * handler ever runs) shape `detail` as a list of `{loc, msg, type}`
 * issues — not the plain string this app's own `ValidationError`/etc.
 * always produce. Turns that list into one readable line per issue
 * (e.g. "feature: Field required") instead of silently discarding it.
 */
function formatValidationIssues(detail) {
  if (!Array.isArray(detail) || detail.length === 0) return null;
  return detail
    .map((issue) => {
      const location = Array.isArray(issue?.loc) ? issue.loc.filter((part) => part !== "body").join(".") : "";
      const message = typeof issue?.msg === "string" ? issue.msg : "Invalid value.";
      return location ? `${location}: ${message}` : message;
    })
    .join("; ");
}

function messageForStatus(status, detail) {
  if (typeof detail === "string" && detail) return detail;
  const validationMessage = formatValidationIssues(detail);
  if (validationMessage) return validationMessage;
  if (status === 404) return "The requested resource could not be found.";
  if (status === 401 || status === 403) return "You are not authorized to perform this action.";
  if (status === 422) return "The request was invalid. Please check your input and try again.";
  if (status >= 500) return "The server encountered an error. Please try again in a moment.";
  return `Request failed with status ${status}.`;
}

async function ensureOk(response) {
  if (response.ok) return response;

  const body = await readErrorBody(response);
  const detail = body?.detail ?? null;
  const message = messageForStatus(response.status, detail);

  // The UI only ever shows `message` (always a short, safe summary),
  // but the complete failure — status, headers, and raw body — always
  // goes to the console too, so diagnosing a real failure never
  // requires reproducing it again (an AI generation call is expensive
  // to just retry for more detail).
  console.error("[apiClient] request failed:", {
    status: response.status,
    statusText: response.statusText,
    headers: Object.fromEntries(response.headers?.entries?.() ?? []),
    body,
  });

  throw new ApiError(message, { status: response.status, detail });
}

/**
 * Wraps `fetch` with a timeout and a single place to translate "the
 * network itself failed" (offline, DNS failure, backend not running,
 * request aborted) into the same `ApiError` shape every caller already
 * handles — callers never need to know the difference between "the
 * server said no" and "we couldn't reach the server at all".
 */
async function fetchWithTimeout(url, options, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new ApiError("The request timed out. Please check your connection and try again.");
    }
    throw new ApiError("Unable to reach the server. Please check your connection and try again.");
  } finally {
    clearTimeout(timeout);
  }
}

async function request(path, options = {}, timeoutMs = undefined) {
  const response = await fetchWithTimeout(
    `${API_BASE_URL}${path}`,
    {
      headers: { "Content-Type": "application/json", ...options.headers },
      ...options,
    },
    timeoutMs,
  );
  await ensureOk(response);
  return response.json();
}

/**
 * For multipart uploads: `body` must be a `FormData` instance, and the
 * `Content-Type` header must be left for the browser to set (it needs
 * to include the multipart boundary) — never forced to
 * `application/json` like every other request.
 */
async function requestForm(path, formData) {
  const response = await fetchWithTimeout(`${API_BASE_URL}${path}`, { method: "POST", body: formData });
  await ensureOk(response);
  return response.json();
}

/**
 * Like `request`, but for endpoints that return a binary file (e.g. an
 * Excel workbook) instead of JSON — resolves to the blob plus whatever
 * filename the server suggested via `Content-Disposition`.
 */
async function requestBlob(path) {
  const response = await fetchWithTimeout(`${API_BASE_URL}${path}`, { method: "GET" });
  await ensureOk(response);

  const blob = await response.blob();
  const contentDisposition = response.headers.get("Content-Disposition");
  return { blob, contentDisposition };
}

export const apiClient = {
  get: (path) => request(path, { method: "GET" }),
  post: (path, body, { timeoutMs } = {}) =>
    request(path, { method: "POST", body: JSON.stringify(body) }, timeoutMs),
  put: (path, body) => request(path, { method: "PUT", body: JSON.stringify(body) }),
  delete: (path) => request(path, { method: "DELETE" }),
  postForm: (path, formData) => requestForm(path, formData),
  getBlob: (path) => requestBlob(path),
};
