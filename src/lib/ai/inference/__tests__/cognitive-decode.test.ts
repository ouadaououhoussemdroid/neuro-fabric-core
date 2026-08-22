/**
 * M33 — Unit tests for the Cognitive State Intelligence decode service.
 *
 * Tests the core `decodeCognitiveState()` logic with a mock ONNX adapter
 * and mock Supabase client. Mirrors the M32 subject-identity test patterns.
 */
import { describe, it, expect, beforeEach, vi, type Mock } from "vitest";

// ─── Mock ONNX adapter (must be at top level — before imports below) ─────

const { mockPredict, mockLoad, mockIsLoaded } = vi.hoisted(() => ({
  mockPredict: vi.fn(),
  mockLoad: vi.fn(),
  mockIsLoaded: vi.fn(),
}));

vi.mock("@/lib/ai/adapters/onnx-adapter", () => {
  return {
    ONNXAdapter: class {
      readonly descriptor: { id: string };
      constructor(opts: { id: string; [key: string]: unknown }) {
        this.descriptor = { id: opts.id };
      }
      predict = mockPredict;
      load = mockLoad;
      isLoaded = mockIsLoaded;
      unload = vi.fn().mockResolvedValue(undefined);
      setModel = vi.fn();
    },
  };
});

// ─── Mock buildServiceProvenance ───────────────────────────────────────────

const mockBuildServiceProvenance = vi.fn();

vi.mock("@/lib/ai/services/provenance.server", () => ({
  buildServiceProvenance: (...args: unknown[]) => mockBuildServiceProvenance(...args),
}));

// ─── Now import the modules under test ──────────────────────────────────────

import {
  decodeCognitiveState,
  type CognitiveDecodeRequest,
  CognitiveDecodeError,
  COGNITIVE_SERVICE,
  COGNITIVE_VERSION,
  COGNITIVE_DEFAULT_HEAD_ID,
  resetCognitiveProbe,
} from "../cognitive.server";
import { JOINT_2312_EMBEDDING_DIM } from "../joint.server";
import { COGNITIVE_LINEAR_PROBE_JOINT_2312 } from "@/lib/ai/decoders/cognitive.registry";

// ─── Mock Supabase client ─────────────────────────────────────────────────

function createMockSupabase(existingEmbedding?: number[]) {
  const embedding = existingEmbedding ?? new Array(JOINT_2312_EMBEDDING_DIM).fill(0).map(() => Math.random());
  return {
    from: vi.fn((table: string) => ({
      select: vi.fn(() => ({
        eq: vi.fn(() => ({
          single: vi.fn().mockResolvedValue({
            data: { embedding },
            error: null,
          }),
        })),
      })),
    })),
    rpc: vi.fn((fn: string, args: Record<string, unknown>) =>
      Promise.resolve({ data: [], error: null }),
    ),
  };
}

// ─── Tests ────────────────────────────────────────────────────────────────

describe("decodeCognitiveState — Cognitive State Intelligence", () => {
  beforeEach(() => {
    resetCognitiveProbe();

    // Default mock behavior
    mockLoad.mockResolvedValue(undefined);
    mockIsLoaded.mockReturnValue(true);
    mockPredict.mockResolvedValue({
      values: { class_0: 0.73 },
      modelId: "cognitive-linear-v1",
      durationMs: 0.42,
    });

    mockBuildServiceProvenance.mockReturnValue({
      service: COGNITIVE_SERVICE,
      service_version: COGNITIVE_VERSION,
      embedding_model: "onnx-cbramod-joint-2312",
      embedding_dim: 2312,
      task_head_id: "cognitive-linear-v1",
      task_head_version: "0.1.0",
      task_head_sha256: "ab8bc6389d98a9461fc7f0f4fea47c3cd9860595c305879351ad0cf6592a6b32",
      task_head_dataset: "PhysioNet EEGMMIDB (workload proxy)",
      task_head_metrics: COGNITIVE_LINEAR_PROBE_JOINT_2312.training?.metrics ?? {
        r2: 0.0, rmse: 0.0, mae: 0.0, pearson_r: 0.0,
      },
      experiment_id: "m33-cognitive-workload-probe",
      timestamp: new Date().toISOString(),
      artifact_shas: {
        cbramod: "c128ccfd00000000000000000000000000000000000000000000000000000000",
        v2: "18644de10000000000000000000000000000000000000000000000000000000",
        pca: "deterministic-pca-v1",
        eegpt: "a92daf440000000000000000000000000000000000000000000000000000000000",
      },
      block_weights: { cbramod: 0.3062, v2: 0.1434, pca: 0.1519, eegpt: 0.3985 },
      component_dims: { cbramod: 200, v2: 32, pca: 32, eegpt: 2048 },
    });
  });

  it("decodes workload from a raw 2312-D query_embedding", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: CognitiveDecodeRequest = { query_embedding: embedding, query_type: "workload" };

    const result = await decodeCognitiveState(req, supabase, "test-user");

    expect(result.service).toBe(COGNITIVE_SERVICE);
    expect(result.head).toBe(COGNITIVE_DEFAULT_HEAD_ID);
    expect(result.results).toHaveLength(1);
    expect(result.results[0].metric).toBe("workload");
    expect(result.results[0].score).toBeGreaterThanOrEqual(0);
    expect(result.results[0].score).toBeLessThanOrEqual(1);
    expect(result.results[0].confidence_interval).toHaveLength(2);
    expect(result.results[0].confidence).toBeGreaterThan(0);
  });

  it("decodes workload from an existing embedding_id (embed-once-reuse)", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.02);
    const supabase = createMockSupabase(embedding);
    const req: CognitiveDecodeRequest = {
      embedding_id: "test-embedding-id",
      query_type: "workload",
    };

    const result = await decodeCognitiveState(req, supabase, "test-user");

    expect(result.metadata.embedding_reused).toBe(true);
    expect(result.embedding_id).toBe("test-embedding-id");
    expect(result.timings.embed_ms).toBeDefined();
  });

  it("throws INVALID_REQUEST when neither embedding_id nor query_embedding provided", async () => {
    const supabase = createMockSupabase();
    const req: CognitiveDecodeRequest = { query_type: "workload" };

    await expect(decodeCognitiveState(req, supabase, "test-user")).rejects.toThrow(
      CognitiveDecodeError,
    );
  });

  it("throws DIMENSION_MISMATCH for wrong embedding dimension", async () => {
    const supabase = createMockSupabase();
    const req: CognitiveDecodeRequest = {
      query_embedding: [0.1, 0.2, 0.3], // too short
      query_type: "workload",
    };

    await expect(decodeCognitiveState(req, supabase, "test-user")).rejects.toThrow(
      "Expected 2312-D embedding, got 3",
    );
  });

  it("throws EMBEDDING_NOT_FOUND when embedding_id doesn't exist", async () => {
    const supabase = {
      from: vi.fn((table: string) => ({
        select: vi.fn(() => ({
          eq: vi.fn(() => ({
            single: vi.fn().mockResolvedValue({
              data: null,
              error: new Error("Not found"),
            }),
          })),
        })),
      })),
      rpc: vi.fn(),
    };
    const req: CognitiveDecodeRequest = {
      embedding_id: "nonexistent-id",
      query_type: "workload",
    };

    await expect(decodeCognitiveState(req, supabase as never, "test-user")).rejects.toThrow(
      "embedding_id not found",
    );
  });

  it("returns provenance with all 4 artifact SHAs", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.03);
    const supabase = createMockSupabase();
    const req: CognitiveDecodeRequest = { query_embedding: embedding };

    const result = await decodeCognitiveState(req, supabase, "test-user");

    expect(result.provenance.artifact_shas.cbramod).toBeDefined();
    expect(result.provenance.artifact_shas.v2).toBeDefined();
    expect(result.provenance.artifact_shas.pca).toBeDefined();
    expect(result.provenance.artifact_shas.eegpt).toBeDefined();
    expect(result.provenance.artifact_shas.cbramod).toMatch(/^c128ccfd/);
    expect(result.provenance.artifact_shas.v2).toMatch(/^18644de1/);
    expect(result.provenance.artifact_shas.eegpt).toMatch(/^a92daf44/);
  });

  it("includes correct service and model in provenance", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.04);
    const supabase = createMockSupabase();
    const req: CognitiveDecodeRequest = { query_embedding: embedding };

    const result = await decodeCognitiveState(req, supabase, "test-user");

    expect(result.provenance.service).toBe(COGNITIVE_SERVICE);
    expect(result.provenance.embedding_model).toBe("onnx-cbramod-joint-2312");
    expect(result.provenance.embedding_dim).toBe(2312);
    expect(result.provenance.task_head_id).toBe("cognitive-linear-v1");
    expect(result.provenance.task_head_metrics?.r2).toBeCloseTo(
      COGNITIVE_LINEAR_PROBE_JOINT_2312.training?.metrics?.r2 ?? 0, 2
    );
    expect(result.provenance.task_head_sha256).toBe(COGNITIVE_LINEAR_PROBE_JOINT_2312.sha256);
  });

  it("includes timings in response", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.05);
    const supabase = createMockSupabase();
    const req: CognitiveDecodeRequest = { query_embedding: embedding };

    const result = await decodeCognitiveState(req, supabase, "test-user");

    expect(result.timings.inference_ms).toBeDefined();
    expect(result.timings.total_ms).toBeDefined();
    expect(typeof result.timings.inference_ms).toBe("number");
  });

  it("returns head_version from registry", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.05);
    const supabase = createMockSupabase();
    const req: CognitiveDecodeRequest = { query_embedding: embedding };

    const result = await decodeCognitiveState(req, supabase, "test-user");

    expect(result.head_version).toBe(COGNITIVE_LINEAR_PROBE_JOINT_2312.version);
  });

  it("supports attention query_type", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.06);
    const supabase = createMockSupabase();
    const req: CognitiveDecodeRequest = {
      query_embedding: embedding,
      query_type: "attention",
    };

    const result = await decodeCognitiveState(req, supabase, "test-user");
    expect(result.results[0].metric).toBe("attention");
  });

  it("supports arousal query_type", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.07);
    const supabase = createMockSupabase();
    const req: CognitiveDecodeRequest = {
      query_embedding: embedding,
      query_type: "arousal",
    };

    const result = await decodeCognitiveState(req, supabase, "test-user");
    expect(result.results[0].metric).toBe("arousal");
  });

  it("default query_type is workload when omitted", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.08);
    const supabase = createMockSupabase();
    const req: CognitiveDecodeRequest = { query_embedding: embedding };

    const result = await decodeCognitiveState(req, supabase, "test-user");
    expect(result.results[0].metric).toBe("workload");
  });

  it("workload score is clamped to [0, 1]", async () => {
    // Override mock to return an out-of-range prediction
    mockPredict.mockResolvedValueOnce({
      values: { class_0: 1.5 }, // out of range
      modelId: "cognitive-linear-v1",
      durationMs: 0.3,
    });

    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.09);
    const supabase = createMockSupabase();
    const req: CognitiveDecodeRequest = { query_embedding: embedding };

    const result = await decodeCognitiveState(req, supabase, "test-user");
    expect(result.results[0].score).toBeLessThanOrEqual(1);
    expect(result.results[0].score).toBeGreaterThanOrEqual(0);
  });
});
