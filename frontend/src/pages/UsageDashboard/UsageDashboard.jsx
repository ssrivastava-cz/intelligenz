import Badge from "../../components/Badge/Badge.jsx";
import Card from "../../components/Card/Card.jsx";
import EmptyState from "../../components/EmptyState/EmptyState.jsx";
import StatCard from "../../components/StatCard/StatCard.jsx";
import Table from "../../components/Table/Table.jsx";
import { useUsageDashboard } from "../../hooks/useUsageDashboard.js";
import { formatDateTime, formatDuration, formatInr, formatSeconds, formatShortDate } from "../../utils/formatters.js";
import "./UsageDashboard.css";

const MISSING_VALUE = "—";

function mono(value) {
  return <span className="data-table__mono">{value}</span>;
}

function formatTokens(tokens) {
  return tokens == null ? MISSING_VALUE : tokens.toLocaleString();
}

// Maps `NormalizedUsageRecordOut.source` onto the badge label/tone the
// unified usage table shows — see backend's `UsageSource`.
const SOURCE_DISPLAY = {
  test_plan_generator: { label: "Test Plan Generator", tone: "default" },
  knowledge_assistant: { label: "Knowledge Assistant", tone: "info" },
};

function SourceBadge({ source }) {
  const display = SOURCE_DISPLAY[source] ?? { label: source, tone: "default" };
  return <Badge tone={display.tone}>{display.label}</Badge>;
}

const USAGE_COLUMNS = [
  { key: "generationId", label: "Generation ID", render: (row) => mono(row.generationId) },
  { key: "source", label: "Source", render: (row) => <SourceBadge source={row.source} /> },
  { key: "model", label: "Model" },
  {
    key: "estimatedInputTokens",
    label: "Estimated Input Tokens",
    render: (row) => formatTokens(row.estimatedInputTokens),
  },
  { key: "inputTokens", label: "Actual Input Tokens", render: (row) => formatTokens(row.inputTokens) },
  { key: "outputTokens", label: "Output Tokens (Billed)", render: (row) => formatTokens(row.outputTokens) },
  {
    key: "reasoningTokens",
    label: "Reasoning Tokens",
    render: (row) => formatTokens(row.reasoningTokens),
  },
  {
    key: "visibleAnswerTokens",
    label: "Visible Answer Tokens",
    render: (row) => formatTokens(row.visibleAnswerTokens),
  },
  {
    key: "structuredOutputOverheadTokens",
    label: "Structured Output Overhead",
    render: (row) => formatTokens(row.structuredOutputOverheadTokens),
  },
  { key: "totalTokens", label: "Total Tokens", render: (row) => formatTokens(row.totalTokens) },
  { key: "inputCostInr", label: "Input Cost", render: (row) => formatInr(row.inputCostInr) },
  { key: "outputCostInr", label: "Output Cost", render: (row) => formatInr(row.outputCostInr) },
  { key: "totalCostInr", label: "Total AI Cost", render: (row) => formatInr(row.totalCostInr) },
  { key: "generationTimeMs", label: "Generation Time", render: (row) => formatDuration(row.generationTimeMs) },
  { key: "createdAt", label: "Created At", render: (row) => formatDateTime(row.createdAt) },
];

const INDEXING_HISTORY_COLUMNS = [
  { key: "date", label: "Date", render: (row) => formatShortDate(row.indexedAt) },
  { key: "feature", label: "Feature" },
  { key: "model", label: "Model", render: (row) => row.embeddingModel },
  { key: "documents", label: "Documents", render: (row) => row.documentsIndexed.toLocaleString() },
  { key: "chunks", label: "Chunks", render: (row) => row.chunksIndexed.toLocaleString() },
  { key: "embeddingTokens", label: "Embedding Tokens", render: (row) => row.embeddingTokens.toLocaleString() },
  { key: "embeddingCost", label: "Embedding Cost", render: (row) => formatInr(row.embeddingCostInr) },
  { key: "indexingTime", label: "Indexing Time", render: (row) => formatSeconds(row.elapsedSeconds) },
  {
    key: "status",
    label: "Status",
    render: (row) => <Badge tone={row.status === "SUCCESS" ? "success" : "danger"}>{row.status}</Badge>,
  },
];

/**
 * Table-only operational view of AI cost — no charts, graphs, or trend
 * visualizations. Covers two separate AI activity types, each with its
 * own stats and table: AI Generation (via `usageApi`, unifying the Test
 * Plan Generator and the Knowledge Assistant into one normalized view —
 * see `useUsageDashboard`) and Document Indexing & Embedding (via
 * `indexingHistoryApi`) — indexing a feature's Source of Truth documents
 * into the knowledge base is a different cost event from a generation,
 * and an indexing run's cost is never folded into a generation's total,
 * or vice versa. Nothing here regenerates AI output, re-embeds
 * anything, or calls OpenAI. The AI Generation Usage table shows every
 * generation from both sources, newest first, tagged with its Source.
 */
function UsageDashboard() {
  const { generations, stats, indexingRuns, indexingStats, isLoading, error } = useUsageDashboard();

  const hasGenerationData = !isLoading && !error && generations.length > 0;
  const isGenerationEmpty = !isLoading && !error && generations.length === 0;
  const hasIndexingData = !isLoading && !error && indexingRuns.length > 0;
  const isIndexingEmpty = !isLoading && !error && indexingRuns.length === 0;
  const isFullyEmpty = isGenerationEmpty && isIndexingEmpty;

  return (
    <div className="usage-dashboard">
      {isLoading && <p className="usage-dashboard__status">Loading usage information…</p>}

      {!isLoading && error && (
        <EmptyState icon="x" title="Unable to load usage information." description="Please try again." />
      )}

      {isFullyEmpty && (
        <EmptyState
          icon="flask"
          title="No AI usage available yet."
          description="Generate your first AI test plan to begin tracking usage."
        />
      )}

      {!isLoading && !error && !isFullyEmpty && (
        <>
          {hasGenerationData && (
            <div className="usage-dashboard__stats">
              <StatCard icon="flask" label="Total Generations" value={stats.totalGenerations.toLocaleString()} />
              <StatCard icon="chart" label="Total Tokens" value={stats.totalTokens.toLocaleString()} />
              <StatCard icon="dashboard" label="Total AI Cost (INR)" value={formatInr(stats.totalAiCostInr)} />
              <StatCard icon="clock" label="Average Cost Per Generation" value={formatInr(stats.averageCostInr)} />
            </div>
          )}
          {isGenerationEmpty && (
            <EmptyState
              icon="flask"
              title="No AI generations yet."
              description="Generate your first AI test plan to begin tracking generation usage."
            />
          )}

          {hasIndexingData && (
            <div className="usage-dashboard__stats">
              <StatCard
                icon="flask"
                label="Total Indexing Runs"
                value={indexingStats.totalIndexingRuns.toLocaleString()}
              />
              <StatCard
                icon="file"
                label="Total Documents Indexed"
                value={indexingStats.totalDocumentsIndexed.toLocaleString()}
              />
              <StatCard
                icon="copy"
                label="Total Chunks Indexed"
                value={indexingStats.totalChunksIndexed.toLocaleString()}
              />
              <StatCard
                icon="chart"
                label="Total Embedding Tokens"
                value={indexingStats.totalEmbeddingTokens.toLocaleString()}
              />
              <StatCard
                icon="dashboard"
                label="Total Embedding Cost (INR)"
                value={formatInr(indexingStats.totalEmbeddingCostInr)}
              />
            </div>
          )}
          {isIndexingEmpty && (
            <EmptyState
              icon="clock"
              title="No indexing runs yet."
              description="Index a feature's Source of Truth documents to begin tracking indexing usage."
            />
          )}

          {hasIndexingData && (
            <Card
              title="Document Indexing & Embedding"
              subtitle="Embedding activity and cost for indexed source documents"
            >
              <Table columns={INDEXING_HISTORY_COLUMNS} rows={indexingRuns} getRowKey={(row) => row.historyId} />
            </Card>
          )}

          {hasGenerationData && (
            <Card
              title="AI Generation Usage"
              subtitle="Combined usage and cost across the Test Plan Generator and Knowledge Assistant, newest first"
            >
              <Table columns={USAGE_COLUMNS} rows={generations} getRowKey={(row) => row.generationId} />
            </Card>
          )}
        </>
      )}
    </div>
  );
}

export default UsageDashboard;
