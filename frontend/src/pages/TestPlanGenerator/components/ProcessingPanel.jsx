import Icon from "../../../components/Icon/Icon.jsx";
import ProgressBar from "../../../components/ProgressBar/ProgressBar.jsx";
import "./ProcessingPanel.css";

/**
 * Animated mock progress view: a progress bar plus a stage-by-stage
 * status list (done / active / pending).
 */
function ProcessingPanel({ stages, currentStageIndex }) {
  const percent = ((currentStageIndex + 1) / stages.length) * 100;

  return (
    <div className="processing-panel">
      <ProgressBar value={percent} />
      <p className="processing-panel__percent">{Math.round(percent)}% complete</p>

      <ul className="processing-panel__stages">
        {stages.map((stage, index) => {
          const state = index < currentStageIndex ? "done" : index === currentStageIndex ? "active" : "pending";

          return (
            <li key={stage} className={`processing-panel__stage processing-panel__stage--${state}`}>
              <span className="processing-panel__stage-icon">
                {state === "done" && <Icon name="checkCircle" size={16} />}
                {state === "active" && <Icon name="spinner" size={16} className="icon-spin" />}
                {state === "pending" && <span className="processing-panel__stage-dot" />}
              </span>
              <span>{stage}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export default ProcessingPanel;
