import { useState } from "react";

import Card from "../../../components/Card/Card.jsx";
import Icon from "../../../components/Icon/Icon.jsx";
import RecentQuestionItem from "./RecentQuestionItem.jsx";
import "./RecentQuestionsSection.css";

/**
 * Collapsible "Recent Questions and Answers" list — collapsed by
 * default so the page doesn't open with a wall of text. Entirely
 * frontend state for now (see `useKnowledgeAssistant`); each item
 * expands independently of this outer toggle.
 * @param {{items: object[]}} props
 */
function RecentQuestionsSection({ items }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <Card
      className="recent-questions-section"
      title="Recent Questions and Answers"
      subtitle={`${items.length} question${items.length === 1 ? "" : "s"} asked recently`}
      actions={
        <button
          type="button"
          className="recent-questions-section__toggle"
          onClick={() => setIsExpanded((previous) => !previous)}
          aria-expanded={isExpanded}
          aria-label={isExpanded ? "Collapse recent questions" : "Expand recent questions"}
        >
          <Icon
            name="chevronDown"
            size={18}
            className={`recent-questions-section__chevron ${isExpanded ? "recent-questions-section__chevron--open" : ""}`.trim()}
          />
        </button>
      }
    >
      {isExpanded && (
        <ul className="recent-questions-section__list">
          {items.map((item) => (
            <RecentQuestionItem key={item.generationId} item={item} />
          ))}
        </ul>
      )}
    </Card>
  );
}

export default RecentQuestionsSection;
