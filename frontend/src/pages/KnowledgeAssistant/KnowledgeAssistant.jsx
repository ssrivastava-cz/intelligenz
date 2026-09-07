import { useEffect, useRef, useState } from "react";

import Button from "../../components/Button/Button.jsx";
import Icon from "../../components/Icon/Icon.jsx";
import { useKnowledgeAssistant } from "../../hooks/useKnowledgeAssistant.js";
import ChatComposer from "./components/ChatComposer.jsx";
import ChatMessage from "./components/ChatMessage.jsx";
import FeedbackModal from "./components/FeedbackModal.jsx";
import "./KnowledgeAssistant.css";

const EXAMPLE_PROMPTS = [
  "How does provider eligibility work?",
  "What are the requirements for appointment eligibility?",
  "How is AI cost calculated?",
];

const SCROLL_BOTTOM_THRESHOLD_PX = 80;

/**
 * ChatGPT-style conversational workspace for product docs, policies,
 * and internal guides — backed by the real Knowledge Assistant backend
 * (`useKnowledgeAssistant`, `POST /knowledge-assistant/ask` and
 * `POST /knowledge-assistant/feedback`), global Source-of-Truth
 * retrieval (no `feature`), unchanged. Conversation history lives only
 * in frontend state for now (see `useKnowledgeAssistant`) — each
 * question is still sent to the backend as an independent request; the
 * backend has no conversation memory yet.
 */
function KnowledgeAssistant() {
  const { messages, isLoading, ask, retry, recordFeedback, startNewConversation } = useKnowledgeAssistant();
  const [feedbackTargetId, setFeedbackTargetId] = useState(null);
  const conversationRef = useRef(null);
  const isPinnedToBottomRef = useRef(true);

  useEffect(() => {
    const conversation = conversationRef.current;
    if (!conversation || !isPinnedToBottomRef.current) return;
    conversation.scrollTop = conversation.scrollHeight;
  }, [messages]);

  function handleConversationScroll(event) {
    const element = event.currentTarget;
    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
    isPinnedToBottomRef.current = distanceFromBottom < SCROLL_BOTTOM_THRESHOLD_PX;
  }

  function handleHelpful(message) {
    recordFeedback(message.generationId, {
      generationId: message.generationId,
      evaluation: "GOOD",
      reason: null,
      description: null,
    });
  }

  function handleNotHelpful(message) {
    setFeedbackTargetId(message.id);
  }

  async function handleFeedbackSubmit({ reason, description }) {
    const target = messages.find((message) => message.id === feedbackTargetId);
    if (!target) return;
    await recordFeedback(target.generationId, {
      generationId: target.generationId,
      evaluation: "BAD",
      reason,
      description,
    });
    setFeedbackTargetId(null);
  }

  return (
    <div className="knowledge-assistant">
      <div className="knowledge-assistant__header">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={startNewConversation}
          disabled={messages.length === 0}
        >
          <Icon name="plus" size={14} />
          New chat
        </Button>
      </div>

      {messages.length === 0 ? (
        <div className="knowledge-assistant__empty">
          <p className="knowledge-assistant__empty-brand">Release Team Intelligenz</p>
          <h2 className="knowledge-assistant__empty-title">How can I help you?</h2>
          <p className="knowledge-assistant__empty-description">
            Ask questions about your product documentation, policies, and guides.
          </p>
          <div className="knowledge-assistant__examples">
            {EXAMPLE_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="knowledge-assistant__example"
                onClick={() => ask(prompt)}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div
          className="knowledge-assistant__conversation"
          ref={conversationRef}
          onScroll={handleConversationScroll}
        >
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              onRetry={retry}
              onHelpful={handleHelpful}
              onNotHelpful={handleNotHelpful}
            />
          ))}
        </div>
      )}

      <div className="knowledge-assistant__composer-dock">
        <ChatComposer onSend={ask} isLoading={isLoading} hasMessages={messages.length > 0} />
      </div>

      <FeedbackModal
        isOpen={feedbackTargetId !== null}
        onCancel={() => setFeedbackTargetId(null)}
        onSubmit={handleFeedbackSubmit}
      />
    </div>
  );
}

export default KnowledgeAssistant;
