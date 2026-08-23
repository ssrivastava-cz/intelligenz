import FileChip from "../../../components/FileChip/FileChip.jsx";
import FileDropzone from "../../../components/FileDropzone/FileDropzone.jsx";
import { ACCEPTED_FILE_ACCEPT_ATTR, ACCEPTED_FILE_HINT } from "../../../config/uploads.js";

/**
 * Upload Documents section: dropzone plus chips for already-added files.
 * @param {{files: object[], addFiles: Function, removeFile: Function, error: string|null}} props
 */
function UploadDocuments({ files, addFiles, removeFile, error }) {
  return (
    <div className="field-group">
      <span className="field-group__label">Upload Documents</span>
      <FileDropzone accept={ACCEPTED_FILE_ACCEPT_ATTR} hint={ACCEPTED_FILE_HINT} onFilesSelected={addFiles} />
      {error && <p className="field-group__error">{error}</p>}
      {files.length > 0 && (
        <div className="upload-documents__chips">
          {files.map((file) => (
            <FileChip key={file.id} file={file} onRemove={removeFile} />
          ))}
        </div>
      )}
    </div>
  );
}

export default UploadDocuments;
