import Button from "../../../components/Button/Button.jsx";
import Card from "../../../components/Card/Card.jsx";
import AnalysisOptions from "./AnalysisOptions.jsx";
import DescriptionField from "./DescriptionField.jsx";
import EmbeddingWorkflow from "./EmbeddingWorkflow.jsx";
import FeatureSelector from "./FeatureSelector.jsx";
import TicketIdField from "./TicketIdField.jsx";
import UploadDocuments from "./UploadDocuments.jsx";
import "./LeftPanel.css";

/**
 * Left panel of the AI Test Plan Generator: all generation inputs plus
 * the Generate action.
 */
function LeftPanel({
  feature,
  onFeatureChange,
  ticketId,
  onTicketIdChange,
  description,
  onDescriptionChange,
  upload,
  isGenerateDisabled,
  onGenerate,
  generateLabel,
}) {
  return (
    <Card className="left-panel" title="Generation Inputs" subtitle="Provide the details for this test plan">
      <div className="left-panel__fields">
        <FeatureSelector value={feature} onChange={onFeatureChange} />
        <TicketIdField value={ticketId} onChange={onTicketIdChange} />
        <DescriptionField value={description} onChange={onDescriptionChange} />
        <UploadDocuments
          files={upload.files}
          addFiles={upload.addFiles}
          removeFile={upload.removeFile}
          error={upload.validationError}
        />
        <EmbeddingWorkflow upload={upload} />
        <AnalysisOptions />
      </div>

      <Button
        variant="primary"
        fullWidth
        disabled={isGenerateDisabled}
        onClick={onGenerate}
        className="left-panel__generate-btn"
      >
        {generateLabel}
      </Button>
    </Card>
  );
}

export default LeftPanel;
