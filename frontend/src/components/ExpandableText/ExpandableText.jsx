import { useState } from "react";
import "./ExpandableText.css";

const DEFAULT_COLLAPSED_LINES = 6;

/**
 * Bordered read-only text panel that clamps long content to a handful
 * of lines with a "Show More" / "Show Less" toggle. Never truncates the
 * underlying data — collapsing is purely visual.
 * @param {{text: string, collapsedLines?: number}} props
 */
function ExpandableText({ text, collapsedLines = DEFAULT_COLLAPSED_LINES }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="expandable-text">
      <p
        className={`expandable-text__content ${isExpanded ? "" : "expandable-text__content--clamped"}`.trim()}
        style={isExpanded ? undefined : { WebkitLineClamp: collapsedLines }}
      >
        {text}
      </p>
      <button
        type="button"
        className="expandable-text__toggle"
        onClick={() => setIsExpanded((previous) => !previous)}
      >
        {isExpanded ? "Show Less" : "Show More"}
      </button>
    </div>
  );
}

export default ExpandableText;
