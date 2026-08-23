import { useState } from "react";

import Button from "../../../components/Button/Button.jsx";
import "./FeedbackSection.css";

/** Thumbs up/down rating plus an optional comment — stored locally only. */
function FeedbackSection() {
  const [rating, setRating] = useState(null);
  const [comment, setComment] = useState("");
  const [submitted, setSubmitted] = useState(false);

  function toggleRating(value) {
    setRating((prev) => (prev === value ? null : value));
    setSubmitted(false);
  }

  function handleSubmit() {
    setSubmitted(true);
  }

  return (
    <div className="feedback-section">
      <h4 className="feedback-section__title">Feedback</h4>
      <div className="feedback-section__ratings">
        <button
          type="button"
          className={`feedback-section__rating-btn ${rating === "up" ? "feedback-section__rating-btn--active-up" : ""}`}
          onClick={() => toggleRating("up")}
        >
          Helpful
        </button>
        <button
          type="button"
          className={`feedback-section__rating-btn ${rating === "down" ? "feedback-section__rating-btn--active-down" : ""}`}
          onClick={() => toggleRating("down")}
        >
          Not Helpful
        </button>
      </div>

      <textarea
        className="feedback-section__comment"
        rows={3}
        placeholder="Add a comment about this test plan (optional)…"
        value={comment}
        onChange={(e) => {
          setComment(e.target.value);
          setSubmitted(false);
        }}
      />

      <div className="feedback-section__footer">
        <Button variant="secondary" size="sm" onClick={handleSubmit} disabled={!rating && !comment.trim()}>
          Send Feedback
        </Button>
        {submitted && <span className="feedback-section__confirmation">Thanks for your feedback!</span>}
      </div>
    </div>
  );
}

export default FeedbackSection;
