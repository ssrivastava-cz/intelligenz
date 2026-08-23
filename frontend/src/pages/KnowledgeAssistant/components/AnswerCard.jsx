import Button from "../../../components/Button/Button.jsx";
import Card from "../../../components/Card/Card.jsx";
import MarkdownText from "../../../components/MarkdownText/MarkdownText.jsx";
import DocumentSourceList from "./DocumentSourceList.jsx";
import "./AnswerCard.css";

/**
 * The answer to the most recently asked question: question, answer,
 * source documents, and the Helpful / Not helpful feedback row. Clicking
 * Helpful records feedback immediately; Not helpful and Add details both
 * defer to the parent-owned `FeedbackModal` (via `onNotHelpful`) so
 * nothing is recorded until the user explicitly submits it there.
 * @param {{record: object, onHelpful: () => void, onNotHelpful: () => void}} props
 */
function AnswerCard({ record, onHelpful, onNotHelpful }) {
  const { question, answer, documents, feedback } = record;
  const isHelpful = feedback?.evaluation === "GOOD";
  const isNotHelpful = feedback?.evaluation === "BAD";
  const hasFeedback = Boolean(feedback);

  return (
    <Card className="answer-card">
      <section className="answer-card__section">
        <p className="answer-card__section-label">Question</p>
        <p className="answer-card__question">{question}</p>
      </section>

      <section className="answer-card__section">
        <p className="answer-card__section-label">Answer</p>
        <MarkdownText text={answer} className="answer-card__answer" />
      </section>

      <section className="answer-card__section">
        <p className="answer-card__section-label">Documents used in this answer</p>
        <DocumentSourceList documents={documents} />
      </section>

      <div className="answer-card__feedback">
        <div className="answer-card__feedback-row">
          <span className="answer-card__feedback-prompt">Was this helpful?</span>
          <Button
            type="button"
            size="sm"
            variant={isHelpful ? "primary" : "secondary"}
            disabled={hasFeedback}
            onClick={onHelpful}
          >
            Helpful
          </Button>
          <Button
            type="button"
            size="sm"
            variant={isNotHelpful ? "danger" : "secondary"}
            disabled={hasFeedback}
            onClick={onNotHelpful}
          >
            Not helpful
          </Button>
          <button type="button" className="answer-card__link" onClick={onNotHelpful}>
            Add details
          </button>
        </div>

        {hasFeedback && <p className="answer-card__thanks">Thanks — your feedback was recorded.</p>}
      </div>
    </Card>
  );
}

export default AnswerCard;
