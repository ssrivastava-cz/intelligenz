import { useState } from "react";

import Card from "../../components/Card/Card.jsx";
import EmptyState from "../../components/EmptyState/EmptyState.jsx";
import StatCard from "../../components/StatCard/StatCard.jsx";
import { generationHistoryApi } from "../../api/generationHistoryApi.js";
import { useGenerationHistory } from "../../hooks/useGenerationHistory.js";
import { formatDuration, formatInr, formatShortDate } from "../../utils/formatters.js";
import GenerationDetailDrawer from "./components/GenerationDetailDrawer.jsx";
import "./GenerationHistory.css";

/**
 * Read-only browser for persisted AI generations: summary stats, a
 * sortable-by-the-backend list, a detail drawer, and an Excel download
 * per row. Never regenerates AI output and never calls OpenAI — every
 * action here only reads (or downloads) already-persisted history.
 */
function GenerationHistory() {
  const { generations, stats, isLoading, error } = useGenerationHistory();
  const [selectedGenerationId, setSelectedGenerationId] = useState(null);
  const [downloadError, setDownloadError] = useState(null);

  function handleDownload(generationId) {
    setDownloadError(null);
    generationHistoryApi.downloadExcel(generationId).catch((caughtError) => {
      setDownloadError(caughtError.message);
    });
  }

  return (
    <div className="generation-history">
      <div className="generation-history__stats">
        <StatCard icon="flask" label="Total Generations" value={(stats?.totalGenerations ?? 0).toLocaleString()} />
        <StatCard icon="chart" label="Total Spend" value={formatInr(stats?.totalSpendInr ?? 0)} />
        <StatCard icon="dashboard" label="Average Cost" value={formatInr(stats?.averageCostInr ?? 0)} />
        <StatCard
          icon="clock"
          label="Avg. Generation Time"
          value={formatDuration(stats?.averageGenerationTimeMs ?? 0)}
        />
      </div>

      <Card title="Test Plan Generation History" subtitle="Every persisted AI test case generation">
        {downloadError && <p className="generation-history__download-error">{downloadError}</p>}

        {isLoading && <p className="generation-history__status">Loading…</p>}

        {!isLoading && error && (
          <EmptyState icon="x" title="Couldn't load test plan generation history" description={error} />
        )}

        {!isLoading && !error && generations.length === 0 && (
          <EmptyState
            icon="flask"
            title="No generations yet"
            description="Generated test plans will appear here once you run the AI Test Plan Generator."
          />
        )}

        {!isLoading && !error && generations.length > 0 && (
          <div className="generation-history__table-wrap">
            <table className="generation-history__table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Feature</th>
                  <th>Redmine Ticket</th>
                  <th>Test Cases</th>
                  <th>Cost (INR)</th>
                  <th>Generation Time</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {generations.map((generation) => (
                  <tr key={generation.generationId}>
                    <td>{formatShortDate(generation.createdAt)}</td>
                    <td>{generation.feature}</td>
                    <td className="generation-history__mono">{generation.redmineTicket}</td>
                    <td>{generation.numberOfTestCases}</td>
                    <td>{formatInr(generation.totalAiCostInr)}</td>
                    <td>{formatDuration(generation.generationTimeMs)}</td>
                    <td className="generation-history__actions">
                      <button
                        type="button"
                        className="generation-history__link"
                        onClick={() => setSelectedGenerationId(generation.generationId)}
                      >
                        View
                      </button>
                      <span className="generation-history__divider" aria-hidden="true">
                        |
                      </span>
                      <button
                        type="button"
                        className="generation-history__link"
                        onClick={() => handleDownload(generation.generationId)}
                      >
                        Download Excel
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <GenerationDetailDrawer generationId={selectedGenerationId} onClose={() => setSelectedGenerationId(null)} />
    </div>
  );
}

export default GenerationHistory;
