import { useEffect, useState } from "react";

import { useDocumentUpload } from "../../hooks/useDocumentUpload.js";
import { useTestPlanGeneration } from "../../hooks/useTestPlanGeneration.js";
import { ACCEPTED_FILE_EXTENSIONS } from "../../config/uploads.js";
import { mapGeneratedTestCases } from "../../utils/mapGeneratedTestCase.js";
import LeftPanel from "./components/LeftPanel.jsx";
import RightPanel from "./components/RightPanel.jsx";
import "./TestPlanGenerator.css";

const GENERATE_LABELS = {
  fetchingTicket: "Fetching ticket details…",
  generating: "Generating test cases…",
  loadingResult: "Finalizing…",
};

/**
 * AI Test Plan Generator — two-panel workspace. The left panel collects
 * generation inputs (and drives Upload -> Embedding Preview -> Embed);
 * the right panel shows idle / processing / results state driven by
 * the real backend pipeline (`useDocumentUpload`, `useTestPlanGeneration`).
 */
function TestPlanGenerator() {
  const [feature, setFeature] = useState("");
  const [ticketId, setTicketId] = useState("");
  const [description, setDescription] = useState("");
  const [rows, setRows] = useState([]);

  const upload = useDocumentUpload(ACCEPTED_FILE_EXTENSIONS);
  const generation = useTestPlanGeneration();

  useEffect(() => {
    if (generation.detail) setRows(mapGeneratedTestCases(generation.detail.testCases));
  }, [generation.detail]);

  const isBusy = ["fetchingTicket", "generating", "loadingResult"].includes(generation.status);
  // Uploading documents without embedding them leaves nothing for
  // retrieval to search — once a session exists, generation waits for
  // that session to actually be embedded (or for the user to remove
  // every file, opting out of uploaded-document context entirely).
  const hasUnembeddedUpload = Boolean(upload.uploadSessionId) && !upload.isEmbedded;
  // The Redmine Ticket ID is optional — Feature is the only required
  // input, so generation can run from it (and retrieved knowledge-base
  // context) alone.
  const isGenerateDisabled = !feature || isBusy || hasUnembeddedUpload;

  function handleGenerate() {
    generation.generate({
      feature,
      ticketId,
      optionalDescription: description,
      uploadSessionId: upload.isEmbedded ? upload.uploadSessionId : null,
    });
  }

  return (
    <div className="test-plan-generator">
      <LeftPanel
        feature={feature}
        onFeatureChange={setFeature}
        ticketId={ticketId}
        onTicketIdChange={setTicketId}
        description={description}
        onDescriptionChange={setDescription}
        upload={upload}
        isGenerateDisabled={isGenerateDisabled}
        onGenerate={handleGenerate}
        generateLabel={GENERATE_LABELS[generation.status] ?? "Generate Test Cases"}
      />
      <RightPanel
        status={generation.status}
        generationStage={generation.generationStage}
        hasTicketId={Boolean(ticketId.trim())}
        error={generation.error}
        detail={generation.detail}
        rows={rows}
        onRowsChange={setRows}
        downloadStatus={generation.downloadStatus}
        downloadError={generation.downloadError}
        onDownloadExcel={generation.downloadExcel}
      />
    </div>
  );
}

export default TestPlanGenerator;
