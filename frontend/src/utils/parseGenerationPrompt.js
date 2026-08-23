/**
 * Recovers the original Redmine Description and User Description text
 * from a persisted generation's `prompt` field.
 *
 * Generation History does not persist these as separate structured
 * fields — PromptBuilder only ever embeds them as plain text inside the
 * full prompt (see backend/app/services/prompt_builder.py). This is a
 * best-effort parse of that fixed, known layout:
 *
 *   ## Feature
 *
 *   Feature Name: <feature>
 *   Redmine Ticket: <ticket>
 *   User Description: <text>        (only present if one was given)
 *
 *   ## Redmine Description
 *
 *   <text>
 *
 *   ## Uploaded Documents
 *   ...
 *
 * It reads back exactly what PromptBuilder already wrote — it does not
 * reconstruct or infer anything — but it is inherently coupled to that
 * layout: if PromptBuilder's section wording ever changes, this parser
 * needs to change with it.
 */

const FEATURE_HEADING = "## Feature";
const REDMINE_DESCRIPTION_HEADING = "## Redmine Description";
const USER_DESCRIPTION_PREFIX = "User Description: ";

/**
 * Returns the raw text between `heading` and the next "\n\n## " section
 * boundary (or the end of the string, if `heading` is the last
 * section), trimmed. Returns null if `heading` isn't found at all.
 */
function extractSection(promptText, heading) {
  const startIndex = promptText.indexOf(heading);
  if (startIndex === -1) return null;

  const afterHeading = promptText.slice(startIndex + heading.length);
  const nextHeadingMatch = afterHeading.match(/\n\n## /);
  const sectionBody = nextHeadingMatch ? afterHeading.slice(0, nextHeadingMatch.index) : afterHeading;
  return sectionBody.trim();
}

/**
 * @param {string | null | undefined} promptText
 * @returns {{redmineDescription: string|null, userDescription: string|null}}
 */
export function parseGenerationQuery(promptText) {
  if (!promptText) {
    return { redmineDescription: null, userDescription: null };
  }

  const redmineDescription = extractSection(promptText, REDMINE_DESCRIPTION_HEADING);

  const featureSection = extractSection(promptText, FEATURE_HEADING);
  let userDescription = null;
  if (featureSection) {
    const prefixIndex = featureSection.indexOf(USER_DESCRIPTION_PREFIX);
    if (prefixIndex !== -1) {
      userDescription = featureSection.slice(prefixIndex + USER_DESCRIPTION_PREFIX.length).trim();
    }
  }

  return { redmineDescription, userDescription };
}
