import { describe, it, expect } from "vitest";
import { NeuralVectorIndex } from "../neural-index";
import type { IndexedVector } from "../index";

function vec<M = unknown>(id: string, values: number[], meta?: M): IndexedVector<M> {
  return { id, vector: values, meta };
}

describe("NeuralVectorIndex (in-memory fallback)", () => {
  it("stores and searches vectors via cosine similarity", async () => {
    const idx = new NeuralVectorIndex<unknown>({});
    await idx.add(vec("a", [1, 0, 0, 0]));
    await idx.add(vec("b", [0, 1, 0, 0]));
    await idx.add(vec("c", [1, 0, 0.1, 0]));

    const hits = await idx.search([1, 0, 0, 0], 2);
    expect(hits).toHaveLength(2);
    expect(hits[0].id).toBe("a");
    expect(hits[0].score).toBeCloseTo(1, 5);
  });

  it("nearest returns the single best match", async () => {
    const idx = new NeuralVectorIndex({});
    await idx.add(vec("a", [1, 0, 0]));
    await idx.add(vec("b", [0, 1, 0]));
    const hit = await idx.nearest([0.9, 0.1, 0]);
    expect(hit?.id).toBe("a");
  });

  it("addAll batch-inserts items", async () => {
    const idx = new NeuralVectorIndex({});
    await idx.addAll([vec("a", [1, 0]), vec("b", [0, 1]), vec("c", [1, 1])]);
    expect(idx.size()).toBe(3);
  });

  it("isPersistent is false without a supabase client", () => {
    const idx = new NeuralVectorIndex({});
    expect(idx.isPersistent).toBe(false);
  });

  it("returns empty array for empty index search", async () => {
    const idx = new NeuralVectorIndex({});
    const hits = await idx.search([1, 0, 0], 5);
    expect(hits).toHaveLength(0);
  });

  it("nearest returns null for empty index", async () => {
    const idx = new NeuralVectorIndex({});
    expect(await idx.nearest([1, 0, 0])).toBeNull();
  });
});

describe("NeuralVectorIndex (pgvector-backed with mock client)", () => {
  it("isPersistent is true when supabase client is provided", () => {
    const mockClient = { from: () => ({}) };
    const idx = new NeuralVectorIndex({ supabase: mockClient });
    expect(idx.isPersistent).toBe(true);
  });

  it("search uses the match_embeddings RPC when available", async () => {
    let rpcCalled = false;
    const mockClient = {
      from: () => ({
        insert: () => ({ select: async () => ({ data: [], error: null }) }),
        select: () => ({ eq: () => ({ limit: () => ({}) }) }),
      }),
      rpc: async (_fn: string, _args: Record<string, unknown>) => {
        rpcCalled = true;
        return {
          data: [
            { id: "hit-1", similarity: 0.95, metadata: { label: "A" } },
            { id: "hit-2", similarity: 0.8, metadata: { label: "B" } },
          ],
          error: null,
        };
      },
    };
    const idx = new NeuralVectorIndex<unknown>({
      supabase: mockClient,
      modelId: "test-model",
      userId: "user-1",
    });
    const hits = await idx.search([1, 0, 0], 5);
    expect(rpcCalled).toBe(true);
    expect(hits).toHaveLength(2);
    expect(hits[0].id).toBe("hit-1");
    expect(hits[0].score).toBe(0.95);
  });

  it("falls back to in-memory when RPC returns an error", async () => {
    const mockClient = {
      from: () => ({
        insert: () => ({ select: async () => ({ data: [], error: null }) }),
      }),
      rpc: async () => ({ data: null, error: { message: "connection failed" } }),
    };
    const idx = new NeuralVectorIndex<Record<string, unknown>>({ supabase: mockClient });
    // Add to in-memory fallback via add (insert succeeds, so it doesn't fall
    // back for storage; but search RPC fails → in-memory search).
    // Since insert doesn't error, the item is in DB (mocked). For the
    // fallback search to find it, we need the in-memory fallback to have it.
    // The add() path only stores in-memory if insert fails. So we expect
    // empty results here (the RPC error path returns empty in-memory search).
    const hits = await idx.search([1, 0, 0], 5);
    expect(hits).toHaveLength(0);
  });
});
