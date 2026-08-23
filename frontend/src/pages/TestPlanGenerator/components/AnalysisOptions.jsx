import Icon from "../../../components/Icon/Icon.jsx";
import { ANALYSIS_OPTIONS } from "../../../config/analysisOptions.js";

/**
 * Static list of analysis steps the (mock) pipeline runs. Informational
 * only — not user-editable.
 */
function AnalysisOptions() {
  return (
    <div className="field-group">
      <span className="field-group__label">Analysis Options</span>
      <ul className="analysis-options">
        {ANALYSIS_OPTIONS.map((option) => (
          <li key={option.id} className="analysis-options__item">
            <Icon name="checkCircle" size={16} className="analysis-options__icon" />
            <span>{option.label}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default AnalysisOptions;
