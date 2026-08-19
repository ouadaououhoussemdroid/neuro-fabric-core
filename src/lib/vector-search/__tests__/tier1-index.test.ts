/**
 * M32 — Tests for DownstreamVectorIndex (Tier-1 result index).
 *
 * Tests the extension of NeuralVectorIndex for downstream Tier-1 result
 * storage. Uses the in-memory fallback (no Supabase client configured),
 * which exercises the same cosine-similarity path as the pgvector RPC.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { DownstreamVectorIndex } from "../tier1-index";

describe("DownstreamVectorIndex — Tier-1 result index", () => {
  let index: DownstreamVectorIndex<{ embedding_id: string }>;

  beforeEach(() => {
    index = new DownstreamVectorIndex<{ embedding_id: string }>({
      service: "subject-identity",
      tableName: "subject_similarity_results",
      matchRpc: "match_subject_similarity",
      matchRpcExact: "match_subject_similarity_exact",
      modelId: "subject-identity-similarity-v1",
      dimensions: 2312,
      searchMode: "ann",
    });
  });

  it("constructs with service name and source table", () => {
    expect(index.service).toBe("subject-identity");
    expect(index.sourceTable).toBe("joint_embeddings_2312");
  });

  it("falls back to in-memory when no Supabase client (isPersistent=false)", () => {
    expect(index.isPersistent).toBe(false);
  });

  it("adds and retrieves items in-memory", async () => {
    const vec = new Array(2312).fill(0);
    vec[0] = 1.0; // unit vector along axis 0
    await index.add({
      id: "item-1",
      vector: vec,
      meta: { embedding_id: "emb-1" },
    });
    // In-memory search should find it
    const hits = await index.search(vec, 1);
    expect(hits.length).toBe(1);
    expect(hits[0].id).toBe("item-1");
    expect(hits[0].score).toBeCloseTo(1.0, 5);
  });

  it("stores wrong-dimension vectors in-memory without error (fallback path)", async () => {
    // When no Supabase client is configured, the in-memory fallback is used.
    // The in-memory VectorIndex does not enforce dimension checks (it just
    // pushes items), matching the existing NeuralVectorIndex behavior.
    // Dimension validation is enforced when a Supabase client is configured
    // (see neural-index.test.ts: "add() to in-memory fallback does NOT validate
    // dimensions"). This test documents that behavior for the Tier-1 index.
    const wrongVec = new Array(200).fill(0);
    wrongVec[0] = 1.0;
    await expect(
      index.add({
        id: "bad-dim",
        vector: wrongVec,
        meta: { embedding_id: "emb-bad" },
      }),
    ).resolves.not.toThrow();
  });

  it("searches and returns sorted results by similarity", async () => {
    // Create 3 vectors: v1 along axis 0, v2 along axis 100, v3 along axis 200
    const v1 = new Array(2312).fill(0);
    v1[0] = 1.0;
    const v2 = new Array(2312).fill(0);
    v2[100] = 1.0;
    const v3 = new Array(2312).fill(0);
    v3[200] = 1.0;

    await index.add({ id: "v1", vector: v1, meta: { embedding_id: "e1" } });
    await index.add({ id: "v2", vector: v2, meta: { embedding_id: "e2" } });
    await index.add({ id: "v3", vector: v3, meta: { embedding_id: "e3" } });

    // Search with v1 — should be closest
    const hits = await index.search(v1, 3);
    expect(hits.length).toBe(3);
    expect(hits[0].id).toBe("v1");
    expect(hits[0].score).toBeCloseTo(1.0, 5);
  });

  it("nearest returns the single closest match", async () => {
    const v1 = new Array(2312).fill(0);
    v1[0] = 1.0;
    const v2 = new Array(2312).fill(0);
    v2[100] = 1.0;

    await index.add({ id: "v1", vector: v1, meta: { embedding_id: "e1" } });
    await index.add({ id: "v2", vector: v2, meta: { embedding_id: "e2" } });

    const nearest = await index.nearest(v1);
    expect(nearest).not.toBeNull();
    expect(nearest!.id).toBe("v1");
  });

  it("nearest returns null when index is empty", async () => {
    const result = await index.nearest(new Array(2312).fill(0));
    expect(result).toBeNull();
  });

  it("size returns the number of stored items", async () => {
    const vec = new Array(2312).fill(0);
    vec[0] = 1.0;
    expect(index.size()).toBe(0);
    await index.add({ id: "a", vector: vec, meta: { embedding_id: "e1" } });
    expect(index.size()).toBe(1);
    await index.add({ id: "b", vector: vec, meta: { embedding_id: "e2" } });
    expect(index.size()).toBe(2);
  });

  it("provenanceMeta tags result with service name", () => {
    const meta = index.provenanceMeta({
      embedding_id: "emb-1",
      model_id: "subject-identity-similarity-v1",
      artifact_shas: {
        cbramod: "sha1",
        v2: "sha2",
        pca: "sha3",
        eegpt: "sha4",
      },
    });
    expect(meta.service).toBe("subject-identity");
    expect(meta.embedding_id).toBe("emb-1");
  });
});
