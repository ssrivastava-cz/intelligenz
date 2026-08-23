import { useCallback, useRef, useState } from "react";

import { generationApi } from "../api/generationApi.js";
import { generationHistoryApi } from "../api/generationHistoryApi.js";
import { redmineApi } from "../api/redmineApi.js";
import { generateId } from "../utils/id.js";

const PROGRESS_POLL_INTERVAL_MS = 400;

/** The Redmine Ticket ID field accepts comma-separated ids; the ticket
 * whose description seeds the generation query is the first one. */
function primaryTicketId(ticketIdInput) {
  return ticketIdInput
    ?.split(",")
    .map((id) => id.trim())
    .find(Boolean);
}

/**
 * Drives AI Test Case Generation: fetches the primary Redmine ticket's
 * description when one is given (a ticket is optional — generation can
 * run from the Feature alone), calls `POST /generate`, then loads the
 * full persisted detail via `GET /generation/{id}` — the same endpoint
 * Generation History already uses — so estimated cost, actual cost, and
 * every generated test case come from one consistent, real record
 * rather than assembling two different response shapes.
 *
 * While `POST /generate` is in flight, this polls `GET
 * /generate/progress/{requestId}` for the real backend stage it's
 * reached (`generationStage`) — never estimated from elapsed time, and
 * never another OpenAI call. Polling only ever reads status; it can't
 * trigger another generation.
 */
export function useTestPlanGeneration() {
  const [status, setStatus] = useState("idle"); // idle | fetchingTicket | generating | loadingResult | success | error
  const [generationStage, setGenerationStage] = useState(null);
  const [error, setError] = useState(null);
  const [generationId, setGenerationId] = useState(null);
  const [detail, setDetail] = useState(null);

  const [downloadStatus, setDownloadStatus] = useState("idle"); // idle | downloading | error
  const [downloadError, setDownloadError] = useState(null);

  const pollHandleRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollHandleRef.current) {
      clearInterval(pollHandleRef.current);
      pollHandleRef.current = null;
    }
  }, []);

  const generate = useCallback(
    async ({ feature, ticketId, optionalDescription, uploadSessionId }) => {
      setError(null);
      setGenerationId(null);
      setDetail(null);
      setGenerationStage(null);

      try {
        let redmineDescription;
        const ticket = primaryTicketId(ticketId);
        if (ticket) {
          setStatus("fetchingTicket");
          const ticketDetail = await redmineApi.getTicketDetail(ticket);
          redmineDescription = ticketDetail.description;
        }

        setStatus("generating");
        const requestId = generateId("gen-request");
        pollHandleRef.current = setInterval(async () => {
          try {
            const progress = await generationApi.getProgress(requestId);
            if (progress.stage) setGenerationStage(progress.stage);
          } catch {
            // Best-effort only — a failed poll never surfaces as a
            // generation error, and never stops the real request.
          }
        }, PROGRESS_POLL_INTERVAL_MS);

        let response;
        try {
          response = await generationApi.generate({
            feature,
            redmineId: ticketId || undefined,
            redmineDescription,
            optionalDescription: optionalDescription || undefined,
            uploadSessionId: uploadSessionId || undefined,
            requestId,
          });
        } finally {
          stopPolling();
        }

        setStatus("loadingResult");
        setGenerationStage("complete");
        const fullDetail = await generationHistoryApi.getById(response.generationId);
        setGenerationId(response.generationId);
        setDetail(fullDetail);
        setStatus("success");
      } catch (caughtError) {
        stopPolling();
        setError(caughtError.message);
        setStatus("error");
      }
    },
    [stopPolling],
  );

  const downloadExcel = useCallback(async () => {
    if (!generationId) return;
    setDownloadStatus("downloading");
    setDownloadError(null);
    try {
      await generationHistoryApi.downloadExcel(generationId);
      setDownloadStatus("idle");
    } catch (caughtError) {
      setDownloadError(caughtError.message);
      setDownloadStatus("error");
    }
  }, [generationId]);

  return {
    status,
    generationStage,
    error,
    generationId,
    detail,
    generate,
    downloadStatus,
    downloadError,
    downloadExcel,
  };
}
