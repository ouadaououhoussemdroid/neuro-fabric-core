/**
 * M32 — Tests for Subject Identity & Cohort Similarity search logic.
 *
 * Tests `searchSubjectIdentity()` with a mock Supabase client (ANN RPC
 * returns pre-computed results), verifying:
 * - Embed-once-reuse: embedding_id path fetches the existing vector
 * - query_embedding path uses the provided vector directly
 * - Confidence computation (gap between top-1 and top-2)
 * - Threshold filtering
 * - Subject exclusion filtering
 * - Error handling (missing embedding, invalid dimensions)
 *
 * Does NOT load real ONNX artifacts — uses synthetic 2312-D vectors.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  searchSubjectIdentity,
  SUBJECT_IDENTITY_MODEL_ID,
  DEFAULT_MATCH_COUNT,
  MAX_MATCH_COUNT,
  SubjectIdentityError,
} from "../subject-identity.server";

// ─────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────

/** Create a synthetic 2312-D unit vector pointing along a single axis. */
function axisVector(axis: number, dim = 2312): number[] {
  const v = new Array(dim).fill(0);
  v[axis] = 1.0;
  return v;
}

/** Create a near-duplicate of a vector (small perturbation along another axis). */
function nearDuplicate(base: number[], jitterAxis: number, jitter: number, dim = 2312): number[] {
  const v = [...base];
  v[jitterAxis] = jitter;
  // Re-normalize to unit length
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map((x) => x / norm);
}

/** Mock Supabase client that simulates the RPC + table query pattern. */
function createMockSupabase(rows: Array<{ id: string; similarity: number; metadata: Record<string, unknown> }>) {
  const store = new Map<string, { embedding: number[] }>();

  return {
    from: (table: string) => ({
      select: (cols: string) => ({
        eq: (col: string, val: string) => ({
          single: () => {
            if (table === "joint_embeddings_2312") {
              const found = store.get(val);
              if (!found) {
                return Promise.resolve({
                  data: null,
                  error: new Error("Not found"),
                });
              }
              return Promise.resolve({
                data: { embedding: found.embedding },
                error: null,
              });
            }
            return Promise.resolve({ data: null, error: null });
          },
        }),
      }),
    }),
    rpc: (fn: string, args: Record<string, unknown>) => {
      const matchCount = (args["match_count"] as number) ?? 10;
      const queryEmbedding = args["query_embedding"] as number[];
      const userId = args["filter_user_id"] as string;

      // Simple cosine similarity against stored rows
      const results = rows.map((r) => {
        // In test, similarity is pre-computed by the test setup
        return r;
      });

      return Promise.resolve({
        data: results.slice(0, matchCount + 1),
        error: null,
      });
    },
    // Internal: store an embedding by id (for embedding_id reuse path)
    __storeEmbedding: (id: string, embedding: number[]) => {
      store.set(id, { embedding });
    },
  };
}

// ─────────────────────────────────────────────────────────────────────

describe("searchSubjectIdentity — Subject Identity & Cohort Similarity", () => {
  const mockEmbeddings = [
    {
      id: "emb-1",
      similarity: 0.95,
      metadata: { subject_id: "S001", session_id: "run5" },
    },
    {
      id: "emb-2",
      similarity: 0.92,
      metadata: { subject_id: "S001", session_id: "run6" },
    },
    {
      id: "emb-3",
      similarity: 0.78,
      metadata: { subject_id: "S002", session_id: "run5" },
    },
    {
      id: "emb-4",
      similarity: 0.65,
      metadata: { subject_id: "S003", session_id: "run5" },
    },
  ];

  it("reuses existing embedding by embedding_id (Embed Once → Reuse Many)", async () => {
    const mockSupabase = createMockSupabase(mockEmbeddings);
    const queryVec = axisVector(0);
    mockSupabase.__storeEmbedding("emb-query", queryVec);

    const response = await searchSubjectIdentity(
      {
        query_type: "subject_identification",
        embedding_id: "emb-query",
        match_count: 5,
        threshold: 0.0, // no threshold filter — return all
      },
      mockSupabase as unknown as Parameters<typeof searchSubjectIdentity>[1],
      "user-1",
    );

    expect(response.results).toBeDefined();
    expect(response.results.length).toBe(4);
    expect(response.metadata.embedding_reused).toBe(true);
    expect(response.results[0].rank).toBe(1);
    expect(response.results[0].similarity).toBe(0.95);
    expect(response.results[0].subject_id).toBe("S001");
  });

  it("uses provided query_embedding directly (no reuse)", async () => {
    const mockSupabase = createMockSupabase(mockEmbeddings);
    const queryVec = axisVector(0);

    const response = await searchSubjectIdentity(
      {
        query_type: "subject_identification",
        query_embedding: queryVec,
        match_count: 5,
      },
      mockSupabase as unknown as Parameters<typeof searchSubjectIdentity>[1],
      "user-1",
    );

    expect(response.metadata.embedding_reused).toBe(false);
  });

  it("computes confidence from gap between top-1 and top-2 similarity", async () => {
    const mockSupabase = createMockSupabase(mockEmbeddings);
    const queryVec = axisVector(0);
    mockSupabase.__storeEmbedding("emb-query", queryVec);

    const response = await searchSubjectIdentity(
      {
        query_type: "subject_identification",
        embedding_id: "emb-query",
        match_count: 5,
      },
      mockSupabase as unknown as Parameters<typeof searchSubjectIdentity>[1],
      "user-1",
    );

    // Top-1 = 0.95, top-2 = 0.92, gap = 0.03
    // confidence = min(1, max(0, 0.03 * 5)) = 0.15
    expect(response.results[0].similarity).toBe(0.95);
    expect(response.results[1].similarity).toBe(0.92);
    expect(response.results[0].confidence).toBeCloseTo(0.15, 2);
  });

  it("throws SubjectIdentityError when neither embedding_id nor query_embedding provided", async () => {
    const mockSupabase = createMockSupabase(mockEmbeddings);

    await expect(
      searchSubjectIdentity(
        { query_type: "subject_identification" },
        mockSupabase as unknown as Parameters<typeof searchSubjectIdentity>[1],
        "user-1",
      ),
    ).rejects.toThrow(SubjectIdentityError);
  });

  it("throws SubjectIdentityError for dimension mismatch (query_embedding)", async () => {
    const mockSupabase = createMockSupabase([]);
    const wrongVec = new Array(200).fill(0);
    wrongVec[0] = 1.0;

    await expect(
      searchSubjectIdentity(
        {
          query_type: "subject_identification",
          query_embedding: wrongVec,
        },
        mockSupabase as unknown as Parameters<typeof searchSubjectIdentity>[1],
        "user-1",
      ),
    ).rejects.toThrow(SubjectIdentityError);
  });

  it("throws SubjectIdentityError when embedding_id not found", async () => {
    const mockSupabase = createMockSupabase([]);

    await expect(
      searchSubjectIdentity(
        {
          query_type: "subject_identification",
          embedding_id: "nonexistent-id",
          match_count: 5,
        },
        mockSupabase as unknown as Parameters<typeof searchSubjectIdentity>[1],
        "user-1",
      ),
    ).rejects.toThrow(SubjectIdentityError);
  });

  it("clamps match_count to MAX_MATCH_COUNT", async () => {
    const mockSupabase = createMockSupabase(mockEmbeddings);
    const queryVec = axisVector(0);
    mockSupabase.__storeEmbedding("emb-query", queryVec);

    const response = await searchSubjectIdentity(
      {
        query_type: "subject_identification",
        embedding_id: "emb-query",
        match_count: 1000, // over limit
      },
      mockSupabase as unknown as Parameters<typeof searchSubjectIdentity>[1],
      "user-1",
    );

    expect(response.metadata.match_count).toBe(MAX_MATCH_COUNT);
  });

  it("applies similarity threshold filter", async () => {
    const mockSupabase = createMockSupabase(mockEmbeddings);
    const queryVec = axisVector(0);
    mockSupabase.__storeEmbedding("emb-query", queryVec);

    const response = await searchSubjectIdentity(
      {
        query_type: "subject_identification",
        embedding_id: "emb-query",
        match_count: 10,
        threshold: 0.80, // only results with similarity >= 0.80
      },
      mockSupabase as unknown as Parameters<typeof searchSubjectIdentity>[1],
      "user-1",
    );

    // 0.95 and 0.92 pass; 0.78 and 0.65 filtered out
    expect(response.results.length).toBe(2);
    expect(response.results.every((r) => r.similarity >= 0.80)).toBe(true);
  });

  it("excludes specified subject_ids", async () => {
    const mockSupabase = createMockSupabase(mockEmbeddings);
    const queryVec = axisVector(0);
    mockSupabase.__storeEmbedding("emb-query", queryVec);

    const response = await searchSubjectIdentity(
      {
        query_type: "subject_identification",
        embedding_id: "emb-query",
        match_count: 10,
        threshold: 0.0, // no threshold filter — return all, then exclude S001
        filter_subject_ids: ["S001"], // exclude S001
      },
      mockSupabase as unknown as Parameters<typeof searchSubjectIdentity>[1],
      "user-1",
    );

    // 0.95 (S001) and 0.92 (S001) should be filtered out
    expect(response.results.length).toBe(2);
    expect(response.results.every((r) => r.subject_id !== "S001")).toBe(true);
  });

  it("returns provenance with correct service and model", async () => {
    const mockSupabase = createMockSupabase(mockEmbeddings);
    const queryVec = axisVector(0);
    mockSupabase.__storeEmbedding("emb-query", queryVec);

    const response = await searchSubjectIdentity(
      {
        query_type: "subject_identification",
        embedding_id: "emb-query",
        threshold: 0.0,
      },
      mockSupabase as unknown as Parameters<typeof searchSubjectIdentity>[1],
      "user-1",
    );

    expect(response.service).toBe("subject-identity");
    expect(response.provenance.service).toBe("subject-identity");
    expect(response.provenance.embedding_model).toBe("onnx-cbramod-joint-2312");
    expect(response.provenance.embedding_dim).toBe(2312);
    expect(response.provenance.task_head_id).toBe(SUBJECT_IDENTITY_MODEL_ID);
    expect(response.provenance.artifact_shas.cbramod).toBe(
      "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
    );
  });

  it("includes timings in response", async () => {
    const mockSupabase = createMockSupabase(mockEmbeddings);
    const queryVec = axisVector(0);
    mockSupabase.__storeEmbedding("emb-query", queryVec);

    const response = await searchSubjectIdentity(
      {
        query_type: "subject_identification",
        embedding_id: "emb-query",
        match_count: 5,
        threshold: 0.0,
      },
      mockSupabase as unknown as Parameters<typeof searchSubjectIdentity>[1],
      "user-1",
    );

    expect(response.timings.search_ms).toBeGreaterThan(0);
    expect(response.timings.total_ms).toBeGreaterThan(0);
  });
});
