import Button from "../../../components/Button/Button.jsx";
import Icon from "../../../components/Icon/Icon.jsx";
import MarkdownText from "../../../components/MarkdownText/MarkdownText.jsx";
import DocumentSourceList from "./DocumentSourceList.jsx";
import TypingIndicator from "./TypingIndicator.jsx";
import "./ChatMessage.css";

const ERROR_FALLBACK = "Sorry, I couldn't retrieve an answer right now. Please try again.";

/**
 * One turn in the conversation. A user message is always plain text
 * (never Markdown-rendered — it's the user's own typed question); an
 * assistant message renders one of three states off `message.status`:
 * `loading` (a neutral `TypingIndicator`, never claiming the LLM is
 * "thinking"), `error` (the failure plus a Retry action, the user's
 * question left untouched above it), or `success` (the real Markdown
 * answer, its source documents exactly as the backend returned them,
 * and the existing Helpful / Not helpful feedback actions).
 * @param {{message: object, onRetry: (id: string) => void, onHelpful: (message: object) => void, onNotHelpful: (message: object) => void}} props
 */
function ChatMessage({ message, onRetry, onHelpful, onNotHelpful }) {
  if (message.role === "user") {
    return (
      <div className="chat-message chat-message--user">
        <div className="chat-message__bubble chat-message__bubble--user">{message.content}</div>
      </div>
    );
  }

  const isHelpful = message.feedback?.evaluation === "GOOD";
  const isNotHelpful = message.feedback?.evaluation === "BAD";
  const hasFeedback = Boolean(message.feedback);

  return (
    <div className="chat-message chat-message--assistant">
      <div className="chat-message__avatar" aria-hidden="true">
        <Icon name="search" size={15} />
      </div>
      <div className="chat-message__body">
        <p className="chat-message__label">Intelligenz</p>
        <div className="chat-message__bubble chat-message__bubble--assistant">
          {message.status === "loading" && <TypingIndicator />}

          {message.status === "error" && (
            <div className="chat-message__error">
              <p className="chat-message__error-text">{message.errorMessage || ERROR_FALLBACK}</p>
              <Button type="button" size="sm" variant="secondary" onClick={() => onRetry(message.id)}>
                Retry
              </Button>
            </div>
          )}

          {message.status === "success" && (
            <>
              <MarkdownText text={message.content} className="chat-message__answer" />

              {message.documents.length > 0 && (
                <div className="chat-message__sources">
                  <p className="chat-message__sources-label">Sources</p>
                  <DocumentSourceList documents={message.documents} />
                </div>
              )}

              <div className="chat-message__feedback">
                <span className="chat-message__feedback-prompt">Was this helpful?</span>
                <Button
                  type="button"
                  size="sm"
                  variant={isHelpful ? "primary" : "ghost"}
                  disabled={hasFeedback}
                  onClick={() => onHelpful(message)}
                >
                  Helpful
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant={isNotHelpful ? "danger" : "ghost"}
                  disabled={hasFeedback}
                  onClick={() => onNotHelpful(message)}
                >
                  Not helpful
                </Button>
                {hasFeedback && <span className="chat-message__thanks">Thanks — your feedback was recorded.</span>}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default ChatMessage;
