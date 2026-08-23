import { useMemo } from "react";

/**
 * Counts words in a text value and flags whether it exceeds a limit.
 * @param {string} text
 * @param {number} limit
 */
export function useWordCount(text, limit = 1500) {
  return useMemo(() => {
    const trimmed = text.trim();
    const wordCount = trimmed === "" ? 0 : trimmed.split(/\s+/).length;
    return { wordCount, limit, isOverLimit: wordCount > limit };
  }, [text, limit]);
}
