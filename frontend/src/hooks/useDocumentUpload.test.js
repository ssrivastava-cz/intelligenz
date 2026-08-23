import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { uploadApi } from "../api/uploadApi.js";
import { useDocumentUpload } from "./useDocumentUpload.js";

vi.mock("../api/uploadApi.js", () => ({
  uploadApi: {
    uploadFiles: vi.fn(),
    getEmbeddingPreview: vi.fn(),
    embedDocuments: vi.fn(),
  },
}));

const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".csv", ".xlsx"];

function makeFile(name) {
  return new File(["content"], name);
}

const UPLOAD_RESPONSE = {
  uploadSessionId: "sess-1",
  documentsUploaded: 1,
  uploadedFiles: [{ documentId: "doc-1", documentName: "notes.txt" }],
};

const PREVIEW_RESPONSE = {
  uploadSessionId: "sess-1",
  documentsFound: 1,
  chunksCreated: 3,
  embeddingModel: "text-embedding-3-small",
  totalEmbeddingTokens: 120,
  averageTokensPerChunk: 40,
  estimatedEmbeddingCost: 0.0001,
  estimatedEmbeddingCostInr: 0.01,
  chunks: [],
};

const EMBED_RESPONSE = {
  uploadSessionId: "sess-1",
  uploadedAt: "2026-08-06T00:00:00Z",
  embeddingModel: "text-embedding-3-small",
  documentsIndexed: 1,
  chunksIndexed: 3,
  averageTokensPerChunk: 40,
  estimatedEmbeddingCost: 0.0001,
  elapsedSeconds: 1.2,
  chromaCollectionName: "uploaded_documents_sess-1",
  indexStatus: "SUCCESS",
};

describe("useDocumentUpload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("rejects files with an unsupported extension without calling the API", async () => {
    const { result } = renderHook(() => useDocumentUpload(ALLOWED_EXTENSIONS));

    act(() => {
      result.current.addFiles([makeFile("virus.exe")]);
    });

    expect(result.current.validationError).toBe("Unsupported file type: virus.exe");
    expect(result.current.files).toEqual([]);
    expect(uploadApi.uploadFiles).not.toHaveBeenCalled();
  });

  it("uploads accepted files and automatically fetches an embedding preview", async () => {
    uploadApi.uploadFiles.mockResolvedValue(UPLOAD_RESPONSE);
    uploadApi.getEmbeddingPreview.mockResolvedValue(PREVIEW_RESPONSE);
    const { result } = renderHook(() => useDocumentUpload(ALLOWED_EXTENSIONS));
    const file = makeFile("notes.txt");

    act(() => {
      result.current.addFiles([file]);
    });

    expect(result.current.uploadStatus).toBe("uploading");
    await waitFor(() => expect(result.current.uploadStatus).toBe("success"));

    expect(uploadApi.uploadFiles).toHaveBeenCalledWith([file]);
    expect(result.current.uploadSessionId).toBe("sess-1");
    expect(result.current.uploadedDocuments).toEqual(UPLOAD_RESPONSE.uploadedFiles);

    await waitFor(() => expect(result.current.previewStatus).toBe("success"));
    expect(uploadApi.getEmbeddingPreview).toHaveBeenCalledWith("sess-1");
    expect(result.current.embeddingPreview).toEqual(PREVIEW_RESPONSE);
  });

  it("sets an error and keeps the staged files when the upload call fails", async () => {
    uploadApi.uploadFiles.mockRejectedValue(new Error("Unable to reach the server."));
    const { result } = renderHook(() => useDocumentUpload(ALLOWED_EXTENSIONS));

    act(() => {
      result.current.addFiles([makeFile("notes.txt")]);
    });

    await waitFor(() => expect(result.current.uploadStatus).toBe("error"));
    expect(result.current.uploadError).toBe("Unable to reach the server.");
    expect(result.current.files).toHaveLength(1);
  });

  it("retryUpload() re-attempts the upload with the currently staged files", async () => {
    uploadApi.uploadFiles.mockRejectedValueOnce(new Error("network down"));
    uploadApi.getEmbeddingPreview.mockResolvedValue(PREVIEW_RESPONSE);
    const { result } = renderHook(() => useDocumentUpload(ALLOWED_EXTENSIONS));

    act(() => {
      result.current.addFiles([makeFile("notes.txt")]);
    });
    await waitFor(() => expect(result.current.uploadStatus).toBe("error"));

    uploadApi.uploadFiles.mockResolvedValue(UPLOAD_RESPONSE);
    await act(async () => {
      result.current.retryUpload();
    });

    await waitFor(() => expect(result.current.uploadStatus).toBe("success"));
    expect(uploadApi.uploadFiles).toHaveBeenCalledTimes(2);
  });

  it("resets the session when the last staged file is removed", async () => {
    uploadApi.uploadFiles.mockResolvedValue(UPLOAD_RESPONSE);
    uploadApi.getEmbeddingPreview.mockResolvedValue(PREVIEW_RESPONSE);
    const { result } = renderHook(() => useDocumentUpload(ALLOWED_EXTENSIONS));

    act(() => {
      result.current.addFiles([makeFile("notes.txt")]);
    });
    await waitFor(() => expect(result.current.uploadStatus).toBe("success"));
    const [staged] = result.current.files;

    act(() => {
      result.current.removeFile(staged.id);
    });

    await waitFor(() => expect(result.current.uploadStatus).toBe("idle"));
    expect(result.current.uploadSessionId).toBeNull();
    expect(result.current.embeddingPreview).toBeNull();
  });

  it("confirmEmbed() embeds the current session and reports the embedded chunk count", async () => {
    uploadApi.uploadFiles.mockResolvedValue(UPLOAD_RESPONSE);
    uploadApi.getEmbeddingPreview.mockResolvedValue(PREVIEW_RESPONSE);
    uploadApi.embedDocuments.mockResolvedValue(EMBED_RESPONSE);
    const { result } = renderHook(() => useDocumentUpload(ALLOWED_EXTENSIONS));

    act(() => {
      result.current.addFiles([makeFile("notes.txt")]);
    });
    await waitFor(() => expect(result.current.previewStatus).toBe("success"));

    await act(async () => {
      await result.current.confirmEmbed();
    });

    expect(uploadApi.embedDocuments).toHaveBeenCalledWith("sess-1");
    expect(result.current.isEmbedded).toBe(true);
    expect(result.current.embedResult.chunksIndexed).toBe(3);
  });

  it("confirmEmbed() surfaces a friendly error and does not mark the session embedded on failure", async () => {
    uploadApi.uploadFiles.mockResolvedValue(UPLOAD_RESPONSE);
    uploadApi.getEmbeddingPreview.mockResolvedValue(PREVIEW_RESPONSE);
    uploadApi.embedDocuments.mockRejectedValue(new Error("OpenAI request failed."));
    const { result } = renderHook(() => useDocumentUpload(ALLOWED_EXTENSIONS));

    act(() => {
      result.current.addFiles([makeFile("notes.txt")]);
    });
    await waitFor(() => expect(result.current.previewStatus).toBe("success"));

    await act(async () => {
      await result.current.confirmEmbed();
    });

    expect(result.current.embedStatus).toBe("error");
    expect(result.current.embedError).toBe("OpenAI request failed.");
    expect(result.current.isEmbedded).toBe(false);
  });
});
