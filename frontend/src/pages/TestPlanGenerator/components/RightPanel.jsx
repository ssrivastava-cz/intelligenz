import Card from "../../../components/Card/Card.jsx";
import EmptyState from "../../../components/EmptyState/EmptyState.jsx";
import { buildStageLabels, resolveCurrentStageIndex } from "../generationStages.js";
import GenerationSummary from "./GenerationSummary.jsx";
import ProcessingPanel from "./ProcessingPanel.jsx";
import TestPlanResults from "./TestPlanResults.jsx";
import "./RightPanel.css";

const IN_PROGRESS_STATUSES = new Set(["fetchingTicket", "generating", "loadingResult"]);

/**
 * Right panel of the AI Test Plan Generator. Renders an idle placeholder,
 * the generation pipeline's real progress (from `useTestPlanGeneration`'s
 * `status`/`generationStage` — never a fixed animation or elapsed-time
 * estimate), an error, or the generated results.
 */
function RightPanel({
  status,
  generationStage,
  hasTicketId,
  error,
  detail,
  rows,
  onRowsChange,
  downloadStatus,
  downloadError,
  onDownloadExcel,
}) {
  if (IN_PROGRESS_STATUSES.has(status)) {
    const currentStageKey = status === "fetchingTicket" ? "fetchingTicket" : generationStage;
    return (
      <Card className="right-panel" title="Generating Test Plan">
        <ProcessingPanel
          stages={buildStageLabels(hasTicketId)}
          currentStageIndex={resolveCurrentStageIndex(hasTicketId, currentStageKey)}
        />
      </Card>
    );
  }

  if (status === "error") {
    return (
      <Card className="right-panel">
        <EmptyState icon="x" title="Generation failed" description={error} />
      </Card>
    );
  }

  if (status === "success" && detail) {
    return (
      <div className="right-panel">
        <GenerationSummary detail={detail} />
        <TestPlanResults
          rows={rows}
          onRowsChange={onRowsChange}
          downloadStatus={downloadStatus}
          downloadError={downloadError}
          onDownloadExcel={onDownloadExcel}
        />
      </div>
    );
  }

  return (
    <Card className="right-panel">
      <EmptyState
        icon="flask"
        title="No test plan generated yet"
        description="Fill out the feature and ticket details on the left, then click Generate Test Cases to create a test plan."
      />
    </Card>
  );
}

export default RightPanel;
