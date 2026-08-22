/**
 * M39 — Unit tests for the Sleep Staging decode service.
 *
 * Tests the core `decodeSleepState()` logic with a mock ONNX adapter
 * and mock Supabase client. Mirrors the M33/M34 service test patterns.
 */
import { describe, it, expect, beforeEach, vi, type Mock } from "vitest";
import { randomUUID } from "node:crypto";
import {
  SLEEP_STAGING_PROBE_JOINT_2312,
} from "@/lib/ai/decoders/sleep.registry";

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

// ─── Mock logging ──────────────────────────────────────────────────────────
vi.mock("@/lib/logging", () => ({
  log: vi.fn(),
  startTimer: vi.fn(() => ({ end: vi.fn().mockReturnValue(0.5) })),
}));

// ─── Now import the modules under test ─────────────────────────────────────
import {
  decodeSleepState,
  type SleepDecodeRequest,
  SleepDecodeError,
  SLEEP_SERVICE,
  SLEEP_VERSION,
  SLEEP_DEFAULT_HEAD_ID,
  resetSleepProbe,
  SLEEP_STAGES_5,
} from "../sleep.server";
import { JOINT_2312_EMBEDDING_DIM } from "../joint.server";

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

// ─── Tests ─────────────────────────────────────────────────────────────────
describe("decodeSleepState — Sleep Staging Service", () => {
  beforeEach(() => {
    resetSleepProbe();

    // Default mock behavior
    mockLoad.mockResolvedValue(undefined);
    mockIsLoaded.mockReturnValue(true);

    // Mock ONNX prediction: return 5 logits (class_0 through class_4)
    mockPredict.mockResolvedValue({
      values: {
        class_0: 0.3, // W
        class_1: 0.1, // N1
        class_2: 0.2, // N2
        class_3: 0.8, // N3 (highest → predicted stage)
        class_4: 0.1, // REM
      },
      modelId: "sleep-staging-v1",
      durationMs: 0.52,
    });

    // Use real registry metrics (not hardcoded zeros)
    mockBuildServiceProvenance.mockImplementation((opts) => ({
      service: opts.service,
      service_version: opts.serviceVersion,
      embedding_model: "onnx-cbramod-joint-2312",
      embedding_dim: 2312,
      task_head_id: opts.taskHeadId,
      task_head_version: opts.taskHeadVersion,
      task_head_sha256: opts.taskHeadSha256,
      task_head_dataset: opts.taskHeadDataset,
      task_head_metrics: opts.taskHeadMetrics,
      experiment_id: opts.experimentId,
      timestamp: new Date().toISOString(),
      artifact_shas: {
        cbramod: SLEEP_STAGING_PROBE_JOINT_2312.training?.metrics ? "c128ccfd..." : "placeholder",
        v2: "18644de1...",
        pca: "deterministic-pca-v1",
        eegpt: "a92daf44...",
      },
      block_weights: { cbramod: 0.3062, v2: 0.1434, pca: 0.1519, eegpt: 0.3985 },
      component_dims: { cbramod: 200, v2: 32, pca: 32, eegpt: 2048 },
    }));
  });

  it("decodes sleep stages from a raw 2312-D query_embedding", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepDecodeRequest = {
      query_embedding: embedding,
      query_type: "sleep-stages",
    };

    const result = await decodeSleepState(req, supabase, "test-user");

    expect(result.service).toBe(SLEEP_SERVICE);
    expect(result.model).toBe("onnx-cbramod-joint-2312");
    expect(result.head).toBe(SLEEP_DEFAULT_HEAD_ID);
    expect(result.results).toHaveLength(1);
    expect(result.results[0].stage_id).toBe(3); // N3 has highest logit
    expect(result.results[0].stage).toBe("N3");
    expect(result.results[0].probabilities).toHaveLength(5);
    expect(result.results[0].confidence).toBeGreaterThan(0);
    expect(result.results[0].confidence).toBeLessThanOrEqual(1);
  });

  it("decodes sleep stages from an existing embedding_id (embed-once-reuse)", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.02);
    const supabase = createMockSupabase(embedding);
    const req: SleepDecodeRequest = {
      embedding_id: "test-embedding-id",
      query_type: "sleep-stages",
    };

    const result = await decodeSleepState(req, supabase, "test-user");

    expect(result.metadata.embedding_reused).toBe(true);
    expect(result.embedding_id).toBe("test-embedding-id");
    expect(result.timings.embed_ms).toBeDefined();
  });

  it("does not include embed_ms when embedding is re-embedded", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepDecodeRequest = {
      query_embedding: embedding,
      query_type: "sleep-stages",
    };

    const result = await decodeSleepState(req, supabase, "test-user");

    expect(result.metadata.embedding_reused).toBe(false);
    expect(result.timings.embed_ms).toBeUndefined();
  });

  it("throws INVALID_REQUEST when neither embedding_id nor query_embedding provided", async () => {
    const supabase = createMockSupabase();
    const req: SleepDecodeRequest = { query_type: "sleep-stages" };

    await expect(decodeSleepState(req, supabase, "test-user")).rejects.toThrow(
      SleepDecodeError,
    );
  });

  it("throws DIMENSION_MISMATCH for wrong embedding dimension", async () => {
    const supabase = createMockSupabase();
    const req: SleepDecodeRequest = {
      query_embedding: [0.1, 0.2, 0.3],
      query_type: "sleep-stages",
    };

    await expect(decodeSleepState(req, supabase, "test-user")).rejects.toThrow(
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

    const req: SleepDecodeRequest = {
      embedding_id: "nonexistent-id",
      query_type: "sleep-stages",
    };

    await expect(decodeSleepState(req, supabase, "test-user")).rejects.toMatchObject({
      code: "EMBEDDING_NOT_FOUND",
    });
  });

  it("throws PROBE_UNAVAILABLE when ONNX adapter fails to load", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    mockLoad.mockRejectedValueOnce(new Error("ONNX file not found"));

    const req: SleepDecodeRequest = {
      query_embedding: embedding,
      query_type: "sleep-stages",
    };

    await expect(decodeSleepState(req, supabase, "test-user")).rejects.toMatchObject({
      code: "PROBE_UNAVAILABLE",
    });
  });

  it("returns correct stage for each stage_id mapping", async () => {
    const stages = ["W", "N1", "N2", "N3", "REM"];
    for (let i = 0; i < 5; i++) {
      const values: Record<string, number> = {};
      values[`class_${i}`] = 10; // This class gets all the weight
      for (let j = 0; j < 5; j++) {
        if (j !== i) values[`class_${j}`] = 0;
      }

      mockPredict.mockResolvedValue({
        values,
        modelId: "sleep-staging-v1",
        durationMs: 0.3,
      });

      const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
      const supabase = createMockSupabase();
      const req: SleepDecodeRequest = {
        query_embedding: embedding,
        query_type: "sleep-stages",
      };

      const result = await decodeSleepState(req, supabase, "test-user");
      expect(result.results[0].stage_id).toBe(i);
      expect(result.results[0].stage).toBe(stages[i]);
    }
  });

  it("returns all 5 probabilities summing to ~1", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepDecodeRequest = {
      query_embedding: embedding,
      query_type: "sleep-stages",
    };

    const result = await decodeSleepState(req, supabase, "test-user");
    const probs = result.results[0].probabilities as number[];
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it("includes provenance with all 4 artifact SHAs", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepDecodeRequest = {
      query_embedding: embedding,
      query_type: "sleep-stages",
    };

    const result = await decodeSleepState(req, supabase, "test-user");

    expect(result.provenance).toBeDefined();
    expect(result.provenance.artifact_shas).toHaveProperty("cbramod");
    expect(result.provenance.artifact_shas).toHaveProperty("v2");
    expect(result.provenance.artifact_shas).toHaveProperty("pca");
    expect(result.provenance.artifact_shas).toHaveProperty("eegpt");
  });

  it("includes head_version and probe_sha256 in metadata", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepDecodeRequest = {
      query_embedding: embedding,
      query_type: "sleep-stages",
    };

    const result = await decodeSleepState(req, supabase, "test-user");

    expect(result.head_version).toBe("0.1.0");
    expect(result.metadata.probe_sha256).toBe(
      "33dde2d3801e74cce6ed33e0e83ec072df62ede9e3ca9c0187ba39f0d7673cff",
    );
  });

  it("SLEEP_STAGES_5 has 5 stages in correct order", () => {
    expect(SLEEP_STAGES_5).toEqual(["W", "N1", "N2", "N3", "REM"]);
  });

  it("throws SleepDecodeError with correct code for invalid request", async () => {
    const supabase = createMockSupabase();
    const req: SleepDecodeRequest = { query_type: "sleep-stages" };

    try {
      await decodeSleepState(req, supabase, "test-user");
      expect(false).toBe(true); // should not reach
    } catch (e) {
      expect(e).toBeInstanceOf(SleepDecodeError);
      expect((e as SleepDecodeError).code).toBe("INVALID_REQUEST");
    }
  });

  it("throws INFERENCE_FAILED when ONNX predict fails", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    mockPredict.mockRejectedValueOnce(new Error("ONNX runtime error"));

    const req: SleepDecodeRequest = {
      query_embedding: embedding,
      query_type: "sleep-stages",
    };

    await expect(decodeSleepState(req, supabase, "test-user")).rejects.toMatchObject({
      code: "INFERENCE_FAILED",
    });
  });

  it("returns 200-compatible response shape with all required fields", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepDecodeRequest = {
      query_embedding: embedding,
      query_type: "sleep-stages",
    };

    const result = await decodeSleepState(req, supabase, "test-user");

    expect(result).toHaveProperty("service");
    expect(result).toHaveProperty("model");
    expect(result).toHaveProperty("head");
    expect(result).toHaveProperty("head_version");
    expect(result).toHaveProperty("provenance");
    expect(result).toHaveProperty("results");
    expect(result).toHaveProperty("metadata");
    expect(result).toHaveProperty("timings");
    expect(result.timings).toHaveProperty("inference_ms");
    expect(result.timings).toHaveProperty("total_ms");
    expect(result.metadata).toHaveProperty("embedding_reused");
    expect(result.metadata).toHaveProperty("probe_sha256");
  });
});
