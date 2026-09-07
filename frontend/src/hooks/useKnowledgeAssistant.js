import { useCallback, useState } from "react";

import { knowledgeAssistantApi } from "../api/knowledgeAssistantApi.js";

let nextMessageId = 0;

function generateMessageId() {
  nextMessageId += 1;
  return `msg_${nextMessageId}`;
}

function pendingAssistantMessage(question) {
  return {
    id: generateMessageId(),
    role: "assistant",
    status: "loading",
    question,
    content: "",
    documents: [],
    usage: null,
    feedback: null,
    generationId: null,
    errorMessage: null,
  };
}

/**
 * Drives the Knowledge Assistant's chat conversation against the real
 * backend — one flat `messages` list, alternating `user`/`assistant`
 * entries in the order they occurred, so the UI can render a continuous
 * conversation instead of a single current answer. Each user question
 * still goes through `POST /knowledge-assistant/ask` as its own,
 * independent request (see `knowledgeAssistantApi.ask`, which never
 * sends a `feature` — retrieval is global now) — the backend has no
 * conversation memory yet, so a follow-up's answer is not informed by
 * earlier turns even though they all stay visible here. Conversations
 * live only in this hook's state; there is no fetch-on-mount of past
 * generations and no persistence beyond the current page session (see
 * `startNewConversation`).
 */
export function useKnowledgeAssistant() {
  const [messages, setMessages] = useState([]);

  const isLoading = messages.some((message) => message.status === "loading");

  const settleAssistantMessage = useCallback(async (assistantMessageId, question) => {
    try {
      const response = await knowledgeAssistantApi.ask({ question });
      setMessages((previous) =>
        previous.map((message) =>
          message.id === assistantMessageId
            ? {
                ...message,
                status: "success",
                content: response.answer,
                documents: response.sourceDocuments.map((name) => ({ name })),
                usage: response.usage,
                generationId: response.generationId,
              }
            : message,
        ),
      );
    } catch (caughtError) {
      setMessages((previous) =>
        previous.map((message) =>
          message.id === assistantMessageId
            ? { ...message, status: "error", errorMessage: caughtError.message }
            : message,
        ),
      );
    }
  }, []);

  const ask = useCallback(
    (question) => {
      if (isLoading) return;
      const trimmedQuestion = question.trim();
      if (!trimmedQuestion) return;

      const userMessage = { id: generateMessageId(), role: "user", content: trimmedQuestion };
      const assistantMessage = pendingAssistantMessage(trimmedQuestion);
      setMessages((previous) => [...previous, userMessage, assistantMessage]);

      return settleAssistantMessage(assistantMessage.id, trimmedQuestion);
    },
    [isLoading, settleAssistantMessage],
  );

  const retry = useCallback(
    (assistantMessageId) => {
      if (isLoading) return;
      const message = messages.find((candidate) => candidate.id === assistantMessageId);
      if (!message) return;

      setMessages((previous) =>
        previous.map((candidate) =>
          candidate.id === assistantMessageId
            ? { ...candidate, status: "loading", errorMessage: null }
            : candidate,
        ),
      );

      return settleAssistantMessage(assistantMessageId, message.question);
    },
    [isLoading, messages, settleAssistantMessage],
  );

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
    setMessages((previous) =>
      previous.map((message) => (message.generationId === generationId ? { ...message, feedback } : message)),
    );
  }, []);

  const startNewConversation = useCallback(() => {
    setMessages([]);
  }, []);

  return {
    messages,
    isLoading,
    ask,
    retry,
    recordFeedback,
    startNewConversation,
  };
}
