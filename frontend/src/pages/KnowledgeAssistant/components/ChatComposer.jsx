import { useRef, useState } from "react";

import Button from "../../../components/Button/Button.jsx";
import Icon from "../../../components/Icon/Icon.jsx";
import "./ChatComposer.css";

const FIRST_QUESTION_PLACEHOLDER = "Ask Intelligenz anything about the available documentation...";
const FOLLOW_UP_PLACEHOLDER = "Ask a follow-up question...";
const MAX_HEIGHT_PX = 160;

/**
 * The sticky message composer at the bottom of the conversation. Owns
 * its own draft text — the parent only ever hears about a message once
 * it's actually sent (`onSend`), and the draft clears the moment that
 * happens rather than waiting for the backend to answer. Enter sends;
 * Shift+Enter inserts a newline (standard chat composer behavior).
 * @param {{onSend: (text: string) => void, isLoading: boolean, hasMessages: boolean}} props
 */
function ChatComposer({ onSend, isLoading, hasMessages }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  function resizeToContent(element) {
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, MAX_HEIGHT_PX)}px`;
  }

  function handleChange(event) {
    setValue(event.target.value);
    resizeToContent(event.target);
  }

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || isLoading) return;
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
    onSend(trimmed);
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <div className="chat-composer">
      <textarea
        ref={textareaRef}
        className="chat-composer__input"
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder={hasMessages ? FOLLOW_UP_PLACEHOLDER : FIRST_QUESTION_PLACEHOLDER}
        rows={1}
        aria-label={hasMessages ? "Ask a follow-up question" : "Ask a question"}
      />
      <Button
        type="button"
        className="chat-composer__send"
        onClick={submit}
        disabled={isLoading || !value.trim()}
        aria-label="Send"
      >
        <Icon name="arrowUp" size={16} />
      </Button>
    </div>
  );
}

export default ChatComposer;
