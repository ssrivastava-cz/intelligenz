/**
 * The real execution stages a generation goes through, shared by
 * `useTestPlanGeneration` (which drives `status`/`generationStage`) and
 * `RightPanel` (which turns those into the `ProcessingPanel` props).
 * "Fetching Redmine ticket details" only applies when a ticket id was
 * actually supplied — `buildStageList` leaves it out otherwise, rather
 * than showing a stage nothing is happening for.
 */
export const STAGE_LABELS = {
  fetchingTicket: "Fetching Redmine ticket details",
  retrievingDocuments: "Retrieving relevant documents",
  buildingPrompt: "Building generation prompt",
  generatingTestCases: "Generating test cases",
  validatingResponse: "Validating AI response",
  savingHistory: "Saving generation history",
  complete: "Generation complete",
};

const STAGE_ORDER = Object.keys(STAGE_LABELS);

export function buildStageKeys(hasTicketId) {
  return hasTicketId ? STAGE_ORDER : STAGE_ORDER.filter((key) => key !== "fetchingTicket");
}

export function buildStageLabels(hasTicketId) {
  return buildStageKeys(hasTicketId).map((key) => STAGE_LABELS[key]);
}

export function resolveCurrentStageIndex(hasTicketId, currentKey) {
  const index = buildStageKeys(hasTicketId).indexOf(currentKey);
  return index === -1 ? 0 : index;
}
