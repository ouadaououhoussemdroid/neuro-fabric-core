/**
 * Phase 2A — Integration tests for the NeuralVectorIndex + concept graph.
 *
 * Tests the full data flow: insert embeddings → search via RPC →
 * query provenance via concept graph. Uses mock Supabase clients
 * that simulate the real `match_embeddings` and `get_embedding_provenance`
 * RPCs, so the tests verify the TS interface contract without requiring
 * a live Postgres.
 */
import { describe, it, expect } from "vitest";
import { NeuralVectorIndex, type NeuralVectorIndexOptions } from "../neural-index";
import {
  createSubject,
  createSession,
  createWindow,
  getEmbeddingProvenance,
  type GraphClient,
} from "../../graph/concept-graph";
import type { IndexedVector, SearchHit } from "../index";

// --- Mock Supabase client for NeuralVectorIndex ---

class MockSupabaseClient {
  private embeddings: Map<
    string,
    { id: string; embedding: number[]; model_id: string; metadata: Record<string, unknown> }
  > = new Map();

  from(table: string) {
    if (table === "embeddings") {
      return {
        insert: (row: Record<string, unknown>) => ({
          select: async () => {
            const id = (row.id as string) || `emb-${this.embeddings.size}`;
            this.embeddings.set(id, {
              id,
              embedding: row.embedding as number[],
              model_id: row.model_id as string,
              metadata: row.metadata as Record<string, unknown>,
            });
            return { data: [{ id }], error: null };
          },
        }),
      };
    }
    return {};
  }

  rpc(fn: string, args: Record<string, unknown>) {
    if (fn === "match_embeddings") {
      const query = args.query_embedding as number[];
      const k = (args.match_count as number) || 10;
      const modelFilter = args.filter_model_id as string | null;
      const results = Array.from(this.embeddings.values())
        .filter((e) => !modelFilter || e.model_id === modelFilter)
        .map((e) => {
          // Cosine similarity
          let dot = 0,
            na = 0,
            nb = 0;
          for (let i = 0; i < query.length; i++) {
            dot += query[i] * e.embedding[i];
            na += query[i] * query[i];
            nb += e.embedding[i] * e.embedding[i];
          }
          const sim = dot / (Math.sqrt(na) * Math.sqrt(nb));
          return { id: e.id, similarity: sim, metadata: e.metadata };
        })
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, k);
      return Promise.resolve({ data: results, error: null });
    }
    return Promise.resolve({ data: null, error: null });
  }
}

// --- Mock Graph client ---

class MockGraphClient {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
  private subjects = new Map<string, Record<string, unknown>>();
  private sessions = new Map<string, Record<string, unknown>>();
  private windows = new Map<string, Record<string, unknown>>();

  from(table: string) {
    return {
      insert: (row: Record<string, unknown>) => ({
        select: async () => {
          const id = (row.id as string) || `${table}-${Date.now()}`;
          if (table === "graph_subjects") {
            // Store with camelCase keys for easier lookup later
            this.subjects.set(id, {
              id,
              userId: row.user_id as string,
              subjectCode: row.subject_code as string,
              dataset: row.dataset as string,
              metadata: (row.metadata as Record<string, unknown>) || {},
              createdAt: (row.created_at as string) || new Date().toISOString(),
              path: (row.path as string) || (row.subject_code as string) || "",
            });
          } else if (table === "graph_sessions") {
            this.sessions.set(id, {
              id,
              subjectId: row.subject_id as string,
              sessionCode: row.session_code as string,
              metadata: (row.metadata as Record<string, unknown>) || {},
              createdAt: (row.created_at as string) || new Date().toISOString(),
              path: (row.path as string) || "",
            });
          } else if (table === "graph_windows") {
            this.windows.set(id, {
              id,
              sessionId: row.session_id as string,
              embeddingId: row.embedding_id as string | null,
              windowIndex: row.window_index as number,
              startSample: row.start_sample as number,
              endSample: row.end_sample as number,
              sampleRate: row.sample_rate as number,
              label: row.label as string | null,
              metadata: (row.metadata as Record<string, unknown>) || {},
              createdAt: (row.created_at as string) || new Date().toISOString(),
              path: (row.path as string) || String(row.window_index || 0),
            });
          }
          // Return the stored data in the format that select() expects
          return { data: [this.getTableRecord(table, id)], error: null };
        },
      }),
      select: () => ({
        // This select is used for direct selects (not chained after insert)
        // For our purposes in this test, we don't need to implement this fully
        eq: () => ({
          single: async () => ({ data: null, error: null }),
          limit: async () => ({ data: [], error: null }),
        }),
      }),
    };
  }

  private getTableRecord(table: string, id: string): Record<string, unknown> {
    if (table === "graph_subjects") return this.subjects.get(id) || {};
    if (table === "graph_sessions") return this.sessions.get(id) || {};
    if (table === "graph_windows") return this.windows.get(id) || {};
    return {};
  }

  rpc(fn: string, args: Record<string, unknown>) {
    if (fn === "get_embedding_provenance") {
      const embId = args.p_embedding_id as string;
      // Return data only for known embeddings, empty for unknown
      if (embId === "emb-1") {
        // Return data in the format that the real RPC would return
        // Based on the getEmbeddingProvenance function, it expects:
        // { data: [{...}], error: null } where the inner array has the actual data
        return Promise.resolve({
          data: [
            {
              subject_code: "S01",
              dataset: "bci-iv-2a",
              session_code: "T1",
              window_index: 5,
              labels: ["left_hand"],
            },
          ],
          error: null,
        });
      }
      // For unknown embeddings, return empty result
      return Promise.resolve({ data: [], error: null });
    }
    return Promise.resolve({ data: null, error: null });
  }
}

describe("Phase 2A: NeuralVectorIndex + concept graph integration", () => {
  it("inserts embeddings and searches via match_embeddings RPC", async () => {
    const mockClient = new MockSupabaseClient();
    const idx = new NeuralVectorIndex({
      supabase: mockClient as unknown as NeuralVectorIndexOptions["supabase"],
      modelId: "eegconformer-prod",
      userId: "user-1",
    });

    // Insert 3 embeddings with different labels
    await idx.add({ id: "emb-1", vector: [1, 0, 0, 0], meta: { label: "left" } });
    await idx.add({ id: "emb-2", vector: [0, 1, 0, 0], meta: { label: "right" } });
    await idx.add({ id: "emb-3", vector: [0.9, 0.1, 0, 0], meta: { label: "left" } });

    // Search for nearest to [1, 0, 0, 0]
    const hits: SearchHit[] = await idx.search([1, 0, 0, 0], 2);
    expect(hits).toHaveLength(2);
    expect(hits[0].id).toBe("emb-1"); // Exact match
    expect(hits[0].score).toBeCloseTo(1.0, 4);
    expect(hits[1].id).toBe("emb-3"); // Close match
    expect(hits[1].score).toBeGreaterThan(0.9);
  });

  it("queries concept-graph provenance for an embedding", async () => {
    const graph = new MockGraphClient();

    // Create a subject → session → window → embedding chain
    const subject = await createSubject(graph as unknown as GraphClient, {
      userId: "user-1",
      subjectCode: "S01",
      dataset: "bci-iv-2a",
      metadata: {},
    });
    expect(subject).not.toBeNull();

    const session = await createSession(graph as unknown as GraphClient, {
      subjectId: subject!.id,
      sessionCode: "T1",
      metadata: {},
    });
    expect(session).not.toBeNull();

    const win = await createWindow(graph as unknown as GraphClient, {
      sessionId: session!.id,
      embeddingId: "emb-1",
      windowIndex: 5,
      startSample: 0,
      endSample: 1000,
      sampleRate: 250,
      label: "left_hand",
      metadata: {},
    });
    expect(win).not.toBeNull();

    // Query provenance
    const prov = await getEmbeddingProvenance(graph as unknown as GraphClient, "emb-1");
    expect(prov).not.toBeNull();
    expect(prov!.subjectCode).toBe("S01");
    expect(prov!.sessionCode).toBe("T1");
    expect(prov!.windowIndex).toBe(5);
    expect(prov!.labels).toContain("left_hand");
  });

  it("handles unknown embedding gracefully in provenance query", async () => {
    const graph = new MockGraphClient();
    const prov = await getEmbeddingProvenance(graph as unknown as GraphClient, "unknown-emb");
    expect(prov).toBeNull();
  });
});
