/**
 * Formats a byte count as a short human-readable size string.
 * @param {number} bytes
 * @returns {string}
 */
export function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Formats an ISO date string as a short readable date/time.
 * @param {string} isoString
 * @returns {string}
 */
export function formatDateTime(isoString) {
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return isoString;

  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

/**
 * Formats an ISO date string as a compact "03 Aug" date, for table rows
 * where the year and time aren't needed.
 * @param {string} isoString
 * @returns {string}
 */
export function formatShortDate(isoString) {
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return isoString;

  return date.toLocaleDateString(undefined, { day: "2-digit", month: "short" });
}

/**
 * Formats a duration given in milliseconds as seconds, e.g. "7.2 sec".
 * @param {number} milliseconds
 * @returns {string}
 */
export function formatDuration(milliseconds) {
  const seconds = Number(milliseconds ?? 0) / 1000;
  return `${seconds.toFixed(1)} sec`;
}

/**
 * Formats a duration already given in seconds (e.g. an indexing run's
 * `elapsedSeconds`) as "6.81 sec" — unlike `formatDuration`, which takes
 * milliseconds, this expects a value that's already in seconds and
 * keeps two decimal places instead of one.
 * @param {number} seconds
 * @returns {string}
 */
export function formatSeconds(seconds) {
  return `${Number(seconds ?? 0).toFixed(2)} sec`;
}

/**
 * Formats an amount in Indian Rupees, e.g. "₹2.43".
 * @param {number} amount
 * @returns {string}
 */
export function formatInr(amount) {
  return `₹${Number(amount ?? 0).toFixed(2)}`;
}

/**
 * Formats an amount in US Dollars with extra precision, since AI
 * generation costs are often a small fraction of a dollar.
 * @param {number} amount
 * @returns {string}
 */
export function formatUsd(amount) {
  return `$${Number(amount ?? 0).toFixed(4)}`;
}
