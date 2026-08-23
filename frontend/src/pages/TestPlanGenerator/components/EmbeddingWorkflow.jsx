import Button from "../../../components/Button/Button.jsx";
import Icon from "../../../components/Icon/Icon.jsx";
import { formatInr, formatUsd } from "../../../utils/formatters.js";
import "./EmbeddingWorkflow.css";

function StatusLine({ icon, spin, text }) {
  return (
    <p className="embedding-workflow__status">
      <Icon name={icon} size={16} className={spin ? "icon-spin" : ""} />
      {text}
    </p>
  );
}

function ErrorLine({ text, onRetry }) {
  return (
    <div className="embedding-workflow__error-row">
      <p className="field-group__error">{text}</p>
      <Button variant="secondary" size="sm" onClick={onRetry}>
        Retry
      </Button>
    </div>
  );
}

/**
 * Embedding Preview -> Embed Documents, shown once an upload session
 * exists. Embedding Preview runs automatically (it's free); Embed
 * Documents only ever runs when the user clicks the button below.
 */
function EmbeddingWorkflow({ upload }) {
  if (upload.uploadStatus === "idle") return null;

  return (
    <div className="embedding-workflow">
      {upload.uploadStatus === "uploading" && <StatusLine icon="spinner" spin text="Uploading documents…" />}

      {upload.uploadStatus === "error" && <ErrorLine text={upload.uploadError} onRetry={upload.retryUpload} />}

      {upload.uploadStatus === "success" && (
        <>
          <StatusLine
            icon="checkCircle"
            text={`${upload.uploadedDocuments.length} document${
              upload.uploadedDocuments.length === 1 ? "" : "s"
            } uploaded`}
          />

          {upload.previewStatus === "loading" && (
            <StatusLine icon="spinner" spin text="Generating embedding preview…" />
          )}
          {upload.previewStatus === "error" && <ErrorLine text={upload.previewError} onRetry={upload.retryPreview} />}
          {upload.previewStatus === "success" && upload.embeddingPreview && (
            <div className="embedding-workflow__preview">
              <dl className="embedding-workflow__preview-grid">
                <div>
                  <dt>Documents Found</dt>
                  <dd>{upload.embeddingPreview.documentsFound}</dd>
                </div>
                <div>
                  <dt>Chunks Created</dt>
                  <dd>{upload.embeddingPreview.chunksCreated}</dd>
                </div>
                <div>
                  <dt>Embedding Tokens</dt>
                  <dd>{upload.embeddingPreview.totalEmbeddingTokens.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Estimated Cost</dt>
                  <dd>
                    {formatUsd(upload.embeddingPreview.estimatedEmbeddingCost)} /{" "}
                    {formatInr(upload.embeddingPreview.estimatedEmbeddingCostInr)}
                  </dd>
                </div>
              </dl>

              {upload.embedStatus === "error" && <ErrorLine text={upload.embedError} onRetry={upload.confirmEmbed} />}

              {upload.embedStatus === "success" ? (
                <StatusLine
                  icon="checkCircle"
                  text={`Embedding complete — ${upload.embedResult.chunksIndexed} chunks embedded`}
                />
              ) : (
                upload.embedStatus !== "error" && (
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={upload.embedStatus === "embedding"}
                    onClick={upload.confirmEmbed}
                  >
                    {upload.embedStatus === "embedding" ? "Embedding documents…" : "Confirm & Embed Documents"}
                  </Button>
                )
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default EmbeddingWorkflow;
