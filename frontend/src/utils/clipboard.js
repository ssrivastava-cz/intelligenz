/**
 * Copies a value to the clipboard as pretty-printed JSON.
 * @param {unknown} data
 * @returns {Promise<boolean>} whether the copy succeeded
 */
export async function copyAsJson(data) {
  try {
    await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    return true;
  } catch {
    return false;
  }
}
