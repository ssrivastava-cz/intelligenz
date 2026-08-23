import Icon from "../Icon/Icon.jsx";
import { formatFileSize } from "../../utils/formatters.js";
import "./FileChip.css";

/**
 * Chip representing a single uploaded file, with a remove control.
 * @param {{file: {id: string, name: string, size: number}, onRemove: (id: string) => void}} props
 */
function FileChip({ file, onRemove }) {
  return (
    <div className="file-chip">
      <Icon name="file" size={14} className="file-chip__icon" />
      <span className="file-chip__name" title={file.name}>
        {file.name}
      </span>
      <span className="file-chip__size">{formatFileSize(file.size)}</span>
      <button
        type="button"
        className="file-chip__remove"
        onClick={() => onRemove(file.id)}
        aria-label={`Remove ${file.name}`}
      >
        <Icon name="x" size={12} />
      </button>
    </div>
  );
}

export default FileChip;
