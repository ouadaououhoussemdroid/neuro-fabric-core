import { describe, it, expect } from "vitest";
import {
  getEmbeddingProvenance,
  createSubject,
  createSession,
  createWindow,
  type GraphClient,
  type GraphSubject,
  type GraphSession,
  type GraphWindow,
} from "../concept-graph";

/** Build a mock GraphClient that returns canned data. */
function mockClient(
  insertData: Record<string, unknown> = {},
  rpcData: unknown = null,
  rpcError: unknown = null,
): GraphClient {
  return {
    from: (table: string) => ({
      insert: (row: unknown) => ({
        select: async () => {
          const base = (insertData[table] ?? row) as Record<string, unknown>;
          const data = [{ ...base, id: "gen-uuid" }];
          return { data, error: null };
        },
      }),
      select: () => ({
        eq: () => ({
          single: async () => ({ data: null, error: null }),
          limit: async () => ({ data: [], error: null }),
        }),
      }),
    }),
    rpc: async () => ({ data: rpcData, error: rpcError }),
  };
}

describe("createSubject", () => {
  it("maps the returned row to camelCase GraphSubject", async () => {
    const client = mockClient({
      graph_subjects: {
        user_id: "u1",
        subject_code: "B01",
        dataset: "bci-iv-2a",
        metadata: { age: 25 },
        path: "B01",
        created_at: "2026-07-11T00:00:00Z",
      },
    });
    const subject = await createSubject(client, {
      userId: "u1",
      subjectCode: "B01",
      dataset: "bci-iv-2a",
      metadata: { age: 25 },
    });
    expect(subject).not.toBeNull();
    expect(subject!.subjectCode).toBe("B01");
    expect(subject!.dataset).toBe("bci-iv-2a");
    expect(subject!.userId).toBe("u1");
    expect(subject!.path).toBe("B01");
  });

  it("returns null when the insert errors", async () => {
    const client: GraphClient = {
      from: () => ({
        insert: () => ({ select: async () => ({ data: null, error: { message: "conflict" } }) }),
        select: () => ({
          eq: () => ({
            single: async () => ({ data: null, error: null }),
            limit: async () => ({ data: [], error: null }),
          }),
        }),
      }),
      rpc: async () => ({ data: null, error: null }),
    };
    const subject = await createSubject(client, {
      userId: "u1",
      subjectCode: "B01",
      dataset: "bci-iv-2a",
      metadata: {},
    });
    expect(subject).toBeNull();
  });
});

describe("getEmbeddingProvenance", () => {
  it("returns the provenance chain from the RPC", async () => {
    const client = mockClient({}, [
      {
        subjectCode: "B01",
        dataset: "bci-iv-2a",
        sessionCode: "T1",
        windowIndex: 5,
        labels: ["left_hand"],
      },
    ]);
    const prov = await getEmbeddingProvenance(client, "emb-uuid");
    expect(prov).not.toBeNull();
    expect(prov!.subjectCode).toBe("B01");
    expect(prov!.sessionCode).toBe("T1");
    expect(prov!.windowIndex).toBe(5);
    expect(prov!.labels).toEqual(["left_hand"]);
  });

  it("returns null when the RPC errors", async () => {
    const client = mockClient({}, null, { message: "rpc failed" });
    const prov = await getEmbeddingProvenance(client, "emb-uuid");
    expect(prov).toBeNull();
  });

  it("returns null when the RPC returns empty data", async () => {
    const client = mockClient({}, []);
    const prov = await getEmbeddingProvenance(client, "emb-uuid");
    expect(prov).toBeNull();
  });
});

describe("createWindow", () => {
  it("maps the returned row correctly", async () => {
    const client = mockClient({
      graph_windows: {
        session_id: "ses-1",
        embedding_id: "emb-1",
        window_index: 3,
        start_sample: 0,
        end_sample: 1000,
        sample_rate: 250,
        label: "right_hand",
        metadata: {},
        path: "B01.T1.3",
        created_at: "2026-07-11T00:00:00Z",
      },
    });
    const win = await createWindow(client, {
      sessionId: "ses-1",
      embeddingId: "emb-1",
      windowIndex: 3,
      startSample: 0,
      endSample: 1000,
      sampleRate: 250,
      label: "right_hand",
      metadata: {},
    });
    expect(win).not.toBeNull();
    expect(win!.windowIndex).toBe(3);
    expect(win!.label).toBe("right_hand");
    expect(win!.path).toBe("B01.T1.3");
  });
});
