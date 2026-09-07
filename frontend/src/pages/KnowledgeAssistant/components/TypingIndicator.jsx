import "./TypingIndicator.css";

/**
 * Assistant "still working" indicator — three pulsing dots plus a
 * neutral status line. Never claims the LLM is "thinking": the backend
 * doesn't expose reasoning state, so this only describes what's
 * actually happening (retrieval against the Source of Truth corpus).
 */
function TypingIndicator() {
  return (
    <div className="typing-indicator">
      <span className="typing-indicator__dots" aria-hidden="true">
        <span className="typing-indicator__dot" />
        <span className="typing-indicator__dot" />
        <span className="typing-indicator__dot" />
      </span>
      <span className="typing-indicator__label">Searching the knowledge base…</span>
    </div>
  );
}

export default TypingIndicator;
