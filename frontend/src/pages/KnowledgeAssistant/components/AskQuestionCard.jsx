import Button from "../../../components/Button/Button.jsx";
import Card from "../../../components/Card/Card.jsx";
import Select from "../../../components/Select/Select.jsx";
import { useSourceOfTruthFeatures } from "../../../hooks/useSourceOfTruthFeatures.js";
import "./AskQuestionCard.css";

const PLACEHOLDER = "e.g. How is the AI cost calculated for a test plan generation?";

/**
 * The "Ask a question" input row, plus a Feature selector. `feature`
 * scopes retrieval — `RetrievalService` (reused unchanged from the Test
 * Plan Generator) only ever searches within one feature, so this reuses
 * the same real feature list `FeatureSelector` does, not a second
 * implementation. Submits on both the Ask button and Enter — `onAsk` is
 * only ever called once both a question and a feature are present.
 * @param {{question: string, onQuestionChange: (value: string) => void, feature: string, onFeatureChange: (value: string) => void, onAsk: () => void, isLoading: boolean}} props
 */
function AskQuestionCard({ question, onQuestionChange, feature, onFeatureChange, onAsk, isLoading }) {
  const { features, isLoading: isLoadingFeatures, error: featuresError } = useSourceOfTruthFeatures();

  function handleKeyDown(event) {
    if (event.key === "Enter") onAsk();
  }

  return (
    <Card title="Ask a question" subtitle="Search across product docs, policies, and internal guides.">
      <div className="ask-question-card__feature">
        <Select
          id="knowledge-assistant-feature"
          label="Feature"
          required
          value={feature}
          onChange={onFeatureChange}
          options={features}
          placeholder={
            isLoadingFeatures ? "Loading features…" : featuresError ? "Unable to load features" : "Select a feature…"
          }
        />
      </div>
      <div className="ask-question-card__row">
        <input
          type="text"
          className="ask-question-card__input"
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={PLACEHOLDER}
          aria-label="Ask a question"
        />
        <Button variant="primary" onClick={onAsk} disabled={isLoading || !question.trim() || !feature}>
          {isLoading ? "Asking…" : "Ask"}
        </Button>
      </div>
    </Card>
  );
}

export default AskQuestionCard;
