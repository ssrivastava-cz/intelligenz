import { useCallback, useEffect, useState } from "react";

import { knowledgeAssistantApi } from "../api/knowledgeAssistantApi.js";

const MAX_RECENT_QUESTIONS = 8;

/**
 * Maps one `GET /knowledge-assistant/recent` row into the same record
 * shape `ask()` produces below, so `RecentQuestionsSection`/
 * `RecentQuestionItem` never need to know whether a record came from
 * the initial fetch or a just-asked question. That endpoint's response
 * carries no feedback status (see `KnowledgeAssistantRecentGenerationOut`)
 * — `null` here matches "no feedback recorded this session", exactly
 * like a freshly-asked question.
 */
function toRecord(recentGeneration) {
  return {
    generationId: recentGeneration.generationId,
    question: recentGeneration.question,
    answer: recentGeneration.answer,
    documents: recentGeneration.sourceDocuments.map((name) => ({ name })),
    usage: null,
    feedback: null,
    createdAt: recentGeneration.createdAt,
  };
}

/**
 * Drives the Knowledge Assistant page against the real backend. Every
 * asked question, its answer, and its feedback live in one
 * `generations` list keyed by `generationId` — there's no separate
 * "current answer" record, just a pointer (`currentGenerationId`) into
 * the same list, so a question's answer and feedback can never drift
 * apart. `generationId` always comes from the backend — this hook
 * never invents one. The recent list is seeded from real, persisted
 * generations (`GET /knowledge-assistant/recent`), never mock/seeded
 * data — if that fetch fails, the list simply stays empty rather than
 * falling back to placeholder questions.
 */
export function useKnowledgeAssistant() {
  const [generations, setGenerations] = useState([]);
  const [currentGenerationId, setCurrentGenerationId] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [error, setError] = useState(null);

  useEffect(() => {
    let isCancelled = false;

    knowledgeAssistantApi
      .listRecent()
      .then((recentGenerations) => {
        if (isCancelled) return;
        setGenerations((previous) => [...previous, ...recentGenerations.map(toRecord)]);
      })
      .catch(() => {
        // The full failure is already logged by apiClient itself
        // (console.error) — never fall back to mock/seeded questions,
        // just leave the recent list empty, a state the UI already
        // renders cleanly.
      });

    return () => {
      isCancelled = true;
    };
  }, []);

  const ask = useCallback(async (question, feature) => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || !feature) return;

    setStatus("loading");
    setError(null);

    try {
      const response = await knowledgeAssistantApi.ask({ question: trimmedQuestion, feature });
      const record = {
        generationId: response.generationId,
        question: response.question,
        answer: response.answer,
        documents: response.sourceDocuments.map((name) => ({ name })),
        usage: response.usage,
        feedback: null,
        createdAt: new Date().toISOString(),
      };
      setGenerations((previous) => [record, ...previous].slice(0, MAX_RECENT_QUESTIONS));
      setCurrentGenerationId(record.generationId);
      setStatus("success");
    } catch (caughtError) {
      // Never add a failed request as a successful Recent Question, and
      // never leave a stale answer showing — only `error` drives the UI now.
      setError(caughtError.message);
      setStatus("error");
    }
  }, []);

  const recordFeedback = useCallback(async (generationId, feedback) => {
    try {
      await knowledgeAssistantApi.submitFeedback({
        generationId,
        evaluation: feedback.evaluation,
        reason: feedback.reason,
        description: feedback.description,
      });
    } catch {
      // The full failure is already logged by apiClient itself
      // (console.error) — feedback simply stays unsubmitted here, so
      // the UI keeps showing Helpful/Not helpful as still actionable
      // rather than a false "Thanks" for a submission that didn't land.
      return;
    }
    setGenerations((previous) =>
      previous.map((generation) => (generation.generationId === generationId ? { ...generation, feedback } : generation)),
    );
  }, []);

  const currentAnswer = generations.find((generation) => generation.generationId === currentGenerationId) ?? null;

  return {
    status,
    error,
    currentAnswer,
    recentQuestions: generations,
    ask,
    recordFeedback,
  };
}
