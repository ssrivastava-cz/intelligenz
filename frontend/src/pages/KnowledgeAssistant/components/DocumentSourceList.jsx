import Icon from "../../../components/Icon/Icon.jsx";
import "./DocumentSourceList.css";

/**
 * Renders the source documents behind an answer. `dense` switches
 * between the full-detail row style used in the main answer card
 * (with a reference badge) and the compact chip style used inside a
 * collapsed Recent Questions entry.
 * @param {{documents: {name: string, reference?: string|null}[], dense?: boolean}} props
 */
function DocumentSourceList({ documents, dense = false }) {
  if (!documents || documents.length === 0) return null;

  if (dense) {
    return (
      <ul className="document-source-list document-source-list--dense">
        {documents.map((document) => (
          <li key={document.name} className="document-source-list__chip">
            <Icon name="file" size={13} />
            <span>{document.name}</span>
          </li>
        ))}
      </ul>
    );
  }

  return (
    <ul className="document-source-list">
      {documents.map((document) => (
        <li key={document.name} className="document-source-list__row">
          <Icon name="file" size={16} className="document-source-list__icon" />
          <span className="document-source-list__name">{document.name}</span>
          {document.reference && <span className="document-source-list__reference">{document.reference}</span>}
        </li>
      ))}
    </ul>
  );
}

export default DocumentSourceList;
