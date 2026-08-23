import { useRef, useState } from "react";

import Icon from "../Icon/Icon.jsx";
import "./FileDropzone.css";

/**
 * Drag-and-drop (or click-to-browse) file upload area.
 * @param {{accept: string, hint: string, multiple?: boolean, onFilesSelected: (files: FileList) => void}} props
 */
function FileDropzone({ accept, hint, multiple = true, onFilesSelected }) {
  const [isDragActive, setIsDragActive] = useState(false);
  const inputRef = useRef(null);

  function handleDrop(event) {
    event.preventDefault();
    setIsDragActive(false);
    if (event.dataTransfer.files?.length) {
      onFilesSelected(event.dataTransfer.files);
    }
  }

  function handleInputChange(event) {
    if (event.target.files?.length) {
      onFilesSelected(event.target.files);
    }
    event.target.value = "";
  }

  return (
    <div
      className={`file-dropzone ${isDragActive ? "file-dropzone--active" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragActive(true);
      }}
      onDragLeave={() => setIsDragActive(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
    >
      <Icon name="upload" size={22} className="file-dropzone__icon" />
      <p className="file-dropzone__text">
        <span className="file-dropzone__link">Click to upload</span> or drag and drop
      </p>
      <p className="file-dropzone__hint">{hint}</p>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={handleInputChange}
        className="visually-hidden"
      />
    </div>
  );
}

export default FileDropzone;
