import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "./client.js";
import { uploadApi } from "./uploadApi.js";

vi.mock("./client.js", () => ({
  apiClient: {
    post: vi.fn(),
    postForm: vi.fn(),
  },
}));

describe("uploadApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("uploadFiles() posts every file under the 'files' field of a FormData body", async () => {
    apiClient.postForm.mockResolvedValue({ uploadSessionId: "sess-1" });
    const fileA = new File(["a"], "a.md");
    const fileB = new File(["b"], "b.pdf");

    await uploadApi.uploadFiles([fileA, fileB]);

    expect(apiClient.postForm).toHaveBeenCalledTimes(1);
    const [path, formData] = apiClient.postForm.mock.calls[0];
    expect(path).toBe("/uploads");
    expect(formData.getAll("files")).toEqual([fileA, fileB]);
  });

  it("getEmbeddingPreview() posts to the session's embedding-preview endpoint", async () => {
    apiClient.post.mockResolvedValue({ documentsFound: 1 });

    await uploadApi.getEmbeddingPreview("sess-1");

    expect(apiClient.post).toHaveBeenCalledWith("/uploads/sess-1/embedding-preview");
  });

  it("embedDocuments() posts to the session's embed endpoint", async () => {
    apiClient.post.mockResolvedValue({ chunksIndexed: 4 });

    await uploadApi.embedDocuments("sess-1");

    expect(apiClient.post).toHaveBeenCalledWith("/uploads/sess-1/embed");
  });
});
