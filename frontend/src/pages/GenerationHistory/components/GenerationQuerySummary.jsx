import ExpandableText from "../../../components/ExpandableText/ExpandableText.jsx";
import { parseGenerationQuery } from "../../../utils/parseGenerationPrompt.js";
import "./GenerationQuerySummary.css";

/**
 * Compact "Generation Query" toolbar shown at the top of the Generation
 * Details drawer, so a user can recognize what a past generation was
 * about without opening the generated test cases first. Feature,
 * Redmine Ticket, and Upload Session come straight from the persisted
 * `GET /generation/{id}` response; Redmine/User Description are
 * recovered from that same response's `prompt` field, since Generation
 * History doesn't persist them as separate fields (see
 * parseGenerationPrompt.js). No additional API requests are made.
 * @param {{detail: object}} props
 */
function GenerationQuerySummary({ detail }) {
  const { redmineDescription, userDescription } = parseGenerationQuery(detail.prompt);
  const uploadSessionId = detail.metadata?.uploadSessionId;

  return (
    <section className="generation-query-summary">
      <h4 className="generation-query-summary__title">Generation Query</h4>

      <dl className="generation-query-summary__fields">
        <div>
          <dt>Feature</dt>
          <dd>{detail.feature}</dd>
        </div>
        <div>
          <dt>Redmine Ticket</dt>
          <dd>{detail.redmineTicket}</dd>
        </div>
        {uploadSessionId && (
          <div>
            <dt>Upload Session</dt>
            <dd>{uploadSessionId}</dd>
          </div>
        )}
      </dl>

      <div className="generation-query-summary__block">
        <h5 className="generation-query-summary__block-title">Redmine Description</h5>
        {redmineDescription ? (
          <ExpandableText text={redmineDescription} />
        ) : (
          <p className="generation-query-summary__empty">No Redmine description available.</p>
        )}
      </div>

      <div className="generation-query-summary__block">
        <h5 className="generation-query-summary__block-title">User Description</h5>
        {userDescription ? (
          <ExpandableText text={userDescription} />
        ) : (
          <p className="generation-query-summary__empty">No additional description provided.</p>
        )}
      </div>
    </section>
  );
}

export default GenerationQuerySummary;
