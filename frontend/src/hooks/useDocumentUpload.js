import { useCallback, useState } from "react";

import { uploadApi } from "../api/uploadApi.js";
import { generateId } from "../utils/id.js";

/**
 * Drives the whole Uploaded Document pipeline for the AI Test Plan
 * Generator: Upload Documents -> Embedding Preview -> Embed Documents.
 *
 * Because `POST /uploads` creates a brand-new, immutable session from
 * exactly the files it's given (there's no "add to an existing
 * session" endpoint), every time the locally staged file list changes
 * (a file is added or removed) the *entire current list* is re-uploaded
 * as one new session, replacing whatever session existed before. This
 * keeps `uploadSessionId` always in sync with what's actually shown as
 * chips, without requiring a separate explicit "Upload" button — files
 * are sent as soon as they're selected, exactly like the workflow asks.
 *
 * Embedding Preview runs automatically right after a successful upload
 * (it's free — local token counting only, no OpenAI call). Embedding
 * itself never runs automatically; it only starts when `confirmEmbed()`
 * is called explicitly.
 */
export function useDocumentUpload(allowedExtensions) {
  const [files, setFiles] = useState([]);
  const [validationError, setValidationError] = useState(null);

  const [uploadStatus, setUploadStatus] = useState("idle"); // idle | uploading | success | error
  const [uploadError, setUploadError] = useState(null);
  const [uploadSessionId, setUploadSessionId] = useState(null);
  const [uploadedDocuments, setUploadedDocuments] = useState([]);

  const [previewStatus, setPreviewStatus] = useState("idle"); // idle | loading | success | error
  const [previewError, setPreviewError] = useState(null);
  const [embeddingPreview, setEmbeddingPreview] = useState(null);

  const [embedStatus, setEmbedStatus] = useState("idle"); // idle | embedding | success | error
  const [embedError, setEmbedError] = useState(null);
  const [embedResult, setEmbedResult] = useState(null);

  const resetEmbeddingState = useCallback(() => {
    setPreviewStatus("idle");
    setPreviewError(null);
    setEmbeddingPreview(null);
    setEmbedStatus("idle");
    setEmbedError(null);
    setEmbedResult(null);
  }, []);

  const fetchPreview = useCallback(async (sessionId) => {
    setPreviewStatus("loading");
    setPreviewError(null);
    try {
      const preview = await uploadApi.getEmbeddingPreview(sessionId);
      setEmbeddingPreview(preview);
      setPreviewStatus("success");
    } catch (error) {
      setPreviewError(error.message);
      setPreviewStatus("error");
    }
  }, []);

  const syncUpload = useCallback(
    async (currentFiles) => {
      resetEmbeddingState();

      if (currentFiles.length === 0) {
        setUploadStatus("idle");
        setUploadError(null);
        setUploadSessionId(null);
        setUploadedDocuments([]);
        return;
      }

      setUploadStatus("uploading");
      setUploadError(null);
      try {
        const response = await uploadApi.uploadFiles(currentFiles.map((staged) => staged.file));
        setUploadSessionId(response.uploadSessionId);
        setUploadedDocuments(response.uploadedFiles);
        setUploadStatus("success");
        await fetchPreview(response.uploadSessionId);
      } catch (error) {
        setUploadError(error.message);
        setUploadStatus("error");
      }
    },
    [fetchPreview, resetEmbeddingState],
  );

  const addFiles = useCallback(
    (fileList) => {
      const incoming = Array.from(fileList);
      const accepted = [];
      const rejected = [];

      for (const file of incoming) {
        const extension = `.${file.name.split(".").pop().toLowerCase()}`;
        if (allowedExtensions.includes(extension)) {
          accepted.push({ id: generateId("file"), name: file.name, size: file.size, file });
        } else {
          rejected.push(file.name);
        }
      }

      setValidationError(rejected.length ? `Unsupported file type: ${rejected.join(", ")}` : null);
      if (!accepted.length) return;

      setFiles((prev) => {
        const next = [...prev, ...accepted];
        syncUpload(next);
        return next;
      });
    },
    [allowedExtensions, syncUpload],
  );

  const removeFile = useCallback(
    (id) => {
      setFiles((prev) => {
        const next = prev.filter((file) => file.id !== id);
        syncUpload(next);
        return next;
      });
    },
    [syncUpload],
  );

  const retryUpload = useCallback(() => {
    syncUpload(files);
  }, [files, syncUpload]);

  const retryPreview = useCallback(() => {
    if (uploadSessionId) fetchPreview(uploadSessionId);
  }, [fetchPreview, uploadSessionId]);

  const confirmEmbed = useCallback(async () => {
    if (!uploadSessionId) return;
    setEmbedStatus("embedding");
    setEmbedError(null);
    try {
      const result = await uploadApi.embedDocuments(uploadSessionId);
      setEmbedResult(result);
      setEmbedStatus("success");
    } catch (error) {
      setEmbedError(error.message);
      setEmbedStatus("error");
    }
  }, [uploadSessionId]);

  return {
    files,
    addFiles,
    removeFile,
    validationError,

    uploadStatus,
    uploadError,
    uploadSessionId,
    uploadedDocuments,
    retryUpload,

    previewStatus,
    previewError,
    embeddingPreview,
    retryPreview,

    embedStatus,
    embedError,
    embedResult,
    confirmEmbed,
    isEmbedded: embedStatus === "success",
  };
}
