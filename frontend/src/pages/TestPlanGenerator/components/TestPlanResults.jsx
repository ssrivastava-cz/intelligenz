import { useState } from "react";

import Button from "../../../components/Button/Button.jsx";
import Card from "../../../components/Card/Card.jsx";
import Icon from "../../../components/Icon/Icon.jsx";
import { copyAsJson } from "../../../utils/clipboard.js";
import { generateId } from "../../../utils/id.js";
import FeedbackSection from "./FeedbackSection.jsx";
import TestPlanTable from "./TestPlanTable.jsx";
import "./TestPlanResults.css";

function blankRow() {
  return {
    id: generateId("TC").toUpperCase(),
    feature: "",
    testCase: "",
    preconditions: "",
    steps: [],
    expectedResult: "",
    priority: "Medium",
    type: "Functional",
    source: "Generated",
    automationCandidate: false,
  };
}

/**
 * Generated test plan: editable table (local review/annotation only)
 * plus a real Excel download and feedback actions. Editing rows here
 * never changes what Download Excel produces — that always downloads
 * the actual persisted generation from the backend (`GET
 * /generation/{id}/download/excel`), the same source of truth
 * Generation History's own download uses, and never regenerates
 * anything.
 */
function TestPlanResults({ rows, onRowsChange, downloadStatus, downloadError, onDownloadExcel }) {
  const [copyStatus, setCopyStatus] = useState("idle");

  function updateRow(id, patch) {
    onRowsChange(rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  }

  function deleteRow(id) {
    onRowsChange(rows.filter((row) => row.id !== id));
  }

  function addRow() {
    onRowsChange([...rows, blankRow()]);
  }

  async function handleCopy() {
    const success = await copyAsJson(rows);
    setCopyStatus(success ? "copied" : "error");
    setTimeout(() => setCopyStatus("idle"), 2000);
  }

  const isDownloading = downloadStatus === "downloading";

  return (
    <Card
      title="Generated Test Plan"
      subtitle={`${rows.length} test case${rows.length === 1 ? "" : "s"}`}
      actions={
        <>
          <Button variant="secondary" size="sm" onClick={onDownloadExcel} disabled={isDownloading}>
            <Icon name="download" size={14} />
            {isDownloading ? "Downloading Excel…" : "Download Excel"}
          </Button>
          <Button variant="secondary" size="sm" onClick={handleCopy}>
            <Icon name="copy" size={14} />
            {copyStatus === "copied" ? "Copied!" : copyStatus === "error" ? "Copy failed" : "Copy JSON"}
          </Button>
        </>
      }
    >
      {downloadStatus === "error" && <p className="field-group__error">{downloadError}</p>}

      <TestPlanTable rows={rows} onUpdateRow={updateRow} onDeleteRow={deleteRow} />

      <Button variant="ghost" size="sm" onClick={addRow} className="test-plan-results__add-row">
        <Icon name="plus" size={14} />
        Add Row
      </Button>

      <div className="test-plan-results__feedback">
        <FeedbackSection />
      </div>
    </Card>
  );
}

export default TestPlanResults;
