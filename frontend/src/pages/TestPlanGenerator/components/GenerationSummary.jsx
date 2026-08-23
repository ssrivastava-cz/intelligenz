import Card from "../../../components/Card/Card.jsx";
import { formatDuration, formatInr, formatUsd } from "../../../utils/formatters.js";
import "./GenerationSummary.css";

/**
 * Cost/usage summary for one completed generation — everything here
 * comes from `GET /generation/{id}` (the same record Generation
 * History shows), never recomputed on the frontend.
 */
function GenerationSummary({ detail }) {
  return (
    <Card title="Generation Summary" className="generation-summary">
      <dl className="generation-summary__fields">
        <div>
          <dt>Generation Time</dt>
          <dd>{formatDuration(detail.generationTimeMs)}</dd>
        </div>
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
        <div>
          <dt>Estimated Prompt Cost</dt>
          <dd>
            {formatUsd(detail.estimatedUsage.estimatedInputCost.usd)} /{" "}
            {formatInr(detail.estimatedUsage.estimatedInputCost.inr)}
          </dd>
        </div>
        <div>
          <dt>Actual AI Cost</dt>
          <dd>
            {formatUsd(detail.actualUsage.totalCost.usd)} / {formatInr(detail.actualUsage.totalCost.inr)}
          </dd>
        </div>
      </dl>
    </Card>
  );
}

export default GenerationSummary;
