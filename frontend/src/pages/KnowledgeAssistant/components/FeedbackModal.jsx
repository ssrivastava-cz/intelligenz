import { useState } from "react";

import Button from "../../../components/Button/Button.jsx";
import TextArea from "../../../components/TextArea/TextArea.jsx";
import "./FeedbackModal.css";

const ISSUE_OPTIONS = [
  { value: "INACCURATE", label: "Inaccurate" },
  { value: "MISSING_INFORMATION", label: "Missing information" },
  { value: "NOT_RELEVANT", label: "Not relevant" },
  { value: "OTHER", label: "Other" },
];

/**
 * "Not helpful" follow-up modal. Renders nothing when closed, so it
 * naturally resets its own fields (issue, details) each time it's
 * reopened for a new answer, without an explicit reset effect.
 * Feedback is only ever recorded when `onSubmit` is called — closing
 * via Cancel (or the backdrop) never reports anything.
 * @param {{isOpen: boolean, onCancel: () => void, onSubmit: (feedback: {reason: string, description: string}) => void}} props
 */
function FeedbackModal({ isOpen, onCancel, onSubmit }) {
  const [reason, setReason] = useState(null);
  const [description, setDescription] = useState("");

  if (!isOpen) return null;

  function handleSubmit() {
    onSubmit({ reason, description: description.trim() });
  }

  return (
    <div className="feedback-modal-overlay" onClick={onCancel}>
      <div
        className="feedback-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Tell us more"
        onClick={(event) => event.stopPropagation()}
      >
        <h3 className="feedback-modal__title">Tell us more</h3>
        <p className="feedback-modal__subtitle">Help us improve this answer.</p>

        <div className="feedback-modal__section">
          <p className="feedback-modal__section-label">What was the issue?</p>
          <div className="feedback-modal__issue-options">
            {ISSUE_OPTIONS.map((option) => (
              <Button
                key={option.value}
                type="button"
                size="sm"
                variant={reason === option.value ? "primary" : "secondary"}
                onClick={() => setReason(option.value)}
              >
                {option.label}
              </Button>
            ))}
          </div>
        </div>

        <div className="feedback-modal__section">
          <TextArea
            id="feedback-details"
            label="Additional details"
            value={description}
            onChange={setDescription}
            placeholder="What was missing or incorrect?"
            rows={3}
          />
        </div>

        <div className="feedback-modal__actions">
          <Button type="button" variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="button" variant="primary" disabled={!reason} onClick={handleSubmit}>
            Submit feedback
          </Button>
        </div>
      </div>
    </div>
  );
}

export default FeedbackModal;
