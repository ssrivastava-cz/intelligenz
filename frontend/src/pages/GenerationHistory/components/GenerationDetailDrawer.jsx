import { useEffect, useState } from "react";

import Drawer from "../../../components/Drawer/Drawer.jsx";
import EmptyState from "../../../components/EmptyState/EmptyState.jsx";
import { generationHistoryApi } from "../../../api/generationHistoryApi.js";
import { formatDateTime, formatDuration, formatInr, formatUsd } from "../../../utils/formatters.js";
import GenerationQuerySummary from "./GenerationQuerySummary.jsx";
import "./GenerationDetailDrawer.css";

/**
 * Read-only detail view for one generation, loaded straight from
 * `GET /generation/{id}` — nothing rebuilt or regenerated. The
 * Generation Query toolbar always appears first, so a user can
 * recognize what a past generation was about before scrolling through
 * usage, cost, retrieval, or test case detail. Rendering nothing until
 * `generationId` is set keeps the drawer closed.
 */
function GenerationDetailDrawer({ generationId, onClose }) {
  const [detail, setDetail] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!generationId) {
      setDetail(null);
      setStatus("idle");
      return;
    }

    let cancelled = false;
    setStatus("loading");
    setError(null);

    generationHistoryApi
      .getById(generationId)
      .then((data) => {
        if (cancelled) return;
        setDetail(data);
        setStatus("success");
      })
      .catch((caughtError) => {
        if (cancelled) return;
        setError(caughtError.message);
        setStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [generationId]);

  return (
    <Drawer isOpen={Boolean(generationId)} title="Generation Details" onClose={onClose}>
      {status === "loading" && <p className="generation-detail__status">Loading…</p>}

      {status === "error" && <EmptyState icon="x" title="Couldn't load this generation" description={error} />}

      {status === "success" && detail && (
        <div className="generation-detail">
          <GenerationQuerySummary detail={detail} />

          <section className="generation-detail__section">
            <h4 className="generation-detail__section-title">Generation Metadata</h4>
            <dl className="generation-detail__fields">
              <div>
                <dt>Generation Date</dt>
                <dd>{formatDateTime(detail.timestamp)}</dd>
              </div>
              <div>
                <dt>Generation Time</dt>
                <dd>{formatDuration(detail.generationTimeMs)}</dd>
              </div>
            </dl>
          </section>

          <section className="generation-detail__section">
            <h4 className="generation-detail__section-title">Token Usage</h4>
            <dl className="generation-detail__fields">
              <div>
                <dt>Prompt Tokens</dt>
                <dd>{detail.actualUsage.promptTokens.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Completion Tokens</dt>
                <dd>{detail.actualUsage.completionTokens.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Total Tokens</dt>
                <dd>{detail.actualUsage.totalTokens.toLocaleString()}</dd>
              </div>
            </dl>
          </section>

          <section className="generation-detail__section">
            <h4 className="generation-detail__section-title">Cost Breakdown</h4>
            <dl className="generation-detail__fields">
              <div>
                <dt>Input Cost</dt>
                <dd>
                  {formatUsd(detail.actualUsage.inputCost.usd)} / {formatInr(detail.actualUsage.inputCost.inr)}
                </dd>
              </div>
              <div>
                <dt>Output Cost</dt>
                <dd>
                  {formatUsd(detail.actualUsage.outputCost.usd)} / {formatInr(detail.actualUsage.outputCost.inr)}
                </dd>
              </div>
              <div>
                <dt>Total Cost</dt>
                <dd>
                  {formatUsd(detail.actualUsage.totalCost.usd)} / {formatInr(detail.actualUsage.totalCost.inr)}
                </dd>
              </div>
            </dl>
          </section>

          <section className="generation-detail__section">
            <h4 className="generation-detail__section-title">Retrieval Summary</h4>
            <dl className="generation-detail__fields">
              <div>
                <dt>Workflow Chunks</dt>
                <dd>{detail.retrieval.workflowChunks}</dd>
              </div>
              <div>
                <dt>Historical Test Cases</dt>
                <dd>{detail.retrieval.historicalTestCases}</dd>
              </div>
              <div>
                <dt>Historical Issues</dt>
                <dd>{detail.retrieval.historicalIssues}</dd>
              </div>
              <div>
                <dt>Uploaded Documents</dt>
                <dd>{detail.retrieval.uploadedDocuments}</dd>
              </div>
            </dl>
          </section>

          <section className="generation-detail__section">
            <h4 className="generation-detail__section-title">Generated Test Cases ({detail.testCases.length})</h4>
            <ul className="generation-detail__test-case-list">
              {detail.testCases.map((testCase) => (
                <li key={testCase.testCaseId}>
                  <span className="generation-detail__test-case-id">{testCase.testCaseId}</span>
                  {testCase.testCaseTitle}
                </li>
              ))}
            </ul>
          </section>
        </div>
      )}
    </Drawer>
  );
}

export default GenerationDetailDrawer;
