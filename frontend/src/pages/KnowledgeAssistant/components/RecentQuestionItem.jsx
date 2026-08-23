import { useState } from "react";

import Icon from "../../../components/Icon/Icon.jsx";
import MarkdownText from "../../../components/MarkdownText/MarkdownText.jsx";
import { formatDateTime } from "../../../utils/formatters.js";
import DocumentSourceList from "./DocumentSourceList.jsx";
import "./RecentQuestionItem.css";

/**
 * One row in "Recent Questions and Answers". Collapsed by default;
 * expanding shows the answer and the (compact) documents used — no
 * feedback controls here, those only apply to the live answer above.
 * @param {{item: {question: string, answer: string, documents: object[], createdAt: string}}} props
 */
function RecentQuestionItem({ item }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <li className="recent-question-item">
      <button
        type="button"
        className="recent-question-item__header"
        onClick={() => setIsExpanded((previous) => !previous)}
        aria-expanded={isExpanded}
      >
        <span className="recent-question-item__text">
          <span className="recent-question-item__question">{item.question}</span>
          <span className="recent-question-item__date">{formatDateTime(item.createdAt)}</span>
        </span>
        <Icon
          name="chevronDown"
          size={16}
          className={`recent-question-item__chevron ${isExpanded ? "recent-question-item__chevron--open" : ""}`.trim()}
        />
      </button>

      {isExpanded && (
        <div className="recent-question-item__body">
          <MarkdownText text={item.answer} className="recent-question-item__answer" />
          <DocumentSourceList documents={item.documents} dense />
        </div>
      )}
    </li>
  );
}

export default RecentQuestionItem;
