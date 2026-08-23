/**
 * Generates a reasonably-unique client-side id for list items (files, rows).
 * @param {string} prefix
 * @returns {string}
 */
export function generateId(prefix = "id") {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
