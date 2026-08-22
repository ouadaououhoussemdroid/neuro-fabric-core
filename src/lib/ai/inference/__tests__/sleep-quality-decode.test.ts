/**
 * M40 — Unit tests for the Sleep Quality decode service.
 *
 * Tests the core `decodeSleepQuality()` logic with a mock ONNX adapter
 * and mock Supabase client. Mirrors the M39 sleep-decode test patterns.
 */
import { describe, it, expect, beforeEach, vi, type Mock } from "vitest";
import { SLEEP_QUALITY_PROBE_JOINT_2312 } from "@/lib/ai/decoders/sleep.registry";

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
  decodeSleepQuality,
  type SleepQualityDecodeRequest,
  SleepDecodeError,
  SLEEP_SERVICE,
  SLEEP_VERSION,
  SLEEP_QUALITY_DEFAULT_HEAD_ID,
  resetSleepProbe,
  resetSleepQualityProbe,
} from "../sleep.server";
import { SLEEP_QUALITY_PROBE_JOINT_2312 } from "@/lib/ai/decoders/sleep.registry";
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
describe("decodeSleepQuality — Sleep Quality Service", () => {
  beforeEach(() => {
    resetSleepProbe();
    resetSleepQualityProbe();

    // Default mock behavior
    mockLoad.mockResolvedValue(undefined);
    mockIsLoaded.mockReturnValue(true);

    // Mock ONNX prediction: returns a single class_0 (quality score)
    mockPredict.mockResolvedValue({
      values: {
        class_0: 0.75,
      },
      modelId: "sleep-quality-v1",
      durationMs: 0.45,
    });

    mockBuildServiceProvenance.mockReturnValue({
      service: SLEEP_SERVICE,
      service_version: SLEEP_VERSION,
      embedding_model: "onnx-cbramod-joint-2312",
      embedding_dim: 2312,
      task_head_id: "sleep-quality-v1",
      task_head_version: "0.1.0",
      task_head_sha256: "5fb7400f1f00037b36f10f9eb73297a346903fef48997c3357cb177a47797d4f",
      task_head_dataset: "Sleep-EDF (PhysioNet 1.0.0)",
      task_head_metrics: SLEEP_QUALITY_PROBE_JOINT_2312.training?.metrics ?? { r2: 0.0, rmse: 0.0, mae: 0.0, pearson_r: 0.0 },
      experiment_id: "m40-sleep-quality-probe",
      timestamp: new Date().toISOString(),
      artifact_shas: {
        cbramod: "c128ccfdf00000000000000000000000000000000000000000000000000000000",
        v2: "18644de1000000000000000000000000000000000000000000000000000000000",
        pca: "deterministic-pca-v1",
        eegpt: "a92daf440000000000000000000000000000000000000000000000000000000000",
      },
      block_weights: { cbramod: 0.3062, v2: 0.1434, pca: 0.1519, eegpt: 0.3985 },
      component_dims: { cbramod: 200, v2: 32, pca: 32, eegpt: 2048 },
    });
  });

  it("decodes sleep quality from a raw 2312-D query_embedding", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepQualityDecodeRequest = { query_embedding: embedding, query_type: "sleep-quality" };

    const result = await decodeSleepQuality(req, supabase, "test-user");

    expect(result.service).toBe(SLEEP_SERVICE);
    expect(result.head).toBe("sleep-quality-v1");
    expect(result.results).toHaveLength(1);
    expect(result.results[0].score).toBeCloseTo(0.75, 2);
    expect(result.results[0].confidence).toBeGreaterThan(0);
    expect(result.results[0].confidence_interval).toHaveLength(2);
    expect(result.results[0].metric).toBe("sleep-quality");
  });

  it("clamps score to [0, 1] when raw output is negative", async () => {
    mockPredict.mockResolvedValue({
      values: { class_0: -5 },
      modelId: "sleep-quality-v1",
      durationMs: 0.3,
    });

    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepQualityDecodeRequest = { query_embedding: embedding, query_type: "sleep-quality" };

    const result = await decodeSleepQuality(req, supabase, "test-user");
    expect(result.results[0].score).toBe(0);
    expect(["poor", "fair", "good", "excellent"]).toContain(result.results[0].band);
  });

  it("clamps score to [0, 1] when raw output exceeds 1", async () => {
    mockPredict.mockResolvedValue({
      values: { class_0: 2.5 },
      modelId: "sleep-quality-v1",
      durationMs: 0.3,
    });

    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepQualityDecodeRequest = { query_embedding: embedding, query_type: "sleep-quality" };

    const result = await decodeSleepQuality(req, supabase, "test-user");
    expect(result.results[0].score).toBe(1);
    expect(result.results[0].band).toBe("excellent");
  });

  it("decodes sleep quality from an existing embedding_id (embed-once-reuse)", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.02);
    const supabase = createMockSupabase(embedding);
    const req: SleepQualityDecodeRequest = {
      embedding_id: "test-embedding-id",
      query_type: "sleep-quality",
    };

    const result = await decodeSleepQuality(req, supabase, "test-user");

    expect(result.metadata.embedding_reused).toBe(true);
    expect(result.embedding_id).toBe("test-embedding-id");
    expect(result.timings.embed_ms).toBeDefined();
  });

  it("sets embedding_reused=false for raw embedding decode", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepQualityDecodeRequest = { query_embedding: embedding, query_type: "sleep-quality" };

    const result = await decodeSleepQuality(req, supabase, "test-user");

    expect(result.metadata.embedding_reused).toBe(false);
  });

  it("throws SleepDecodeError with INVALID_REQUEST when neither embedding provided", async () => {
    const supabase = createMockSupabase();
    const req: SleepQualityDecodeRequest = { query_type: "sleep-quality" };

    try {
      await decodeSleepQuality(req, supabase, "test-user");
      expect(false).toBe(true);
    } catch (e) {
      expect(e).toBeInstanceOf(SleepDecodeError);
      expect((e as SleepDecodeError).code).toBe("INVALID_REQUEST");
    }
  });

  it("throws DIMENSION_MISMATCH for wrong embedding dimension", async () => {
    const supabase = createMockSupabase();
    const req: SleepQualityDecodeRequest = {
      query_embedding: [0.1, 0.2, 0.3],
      query_type: "sleep-quality",
    };

    await expect(decodeSleepQuality(req, supabase, "test-user")).rejects.toMatchObject({
      code: "DIMENSION_MISMATCH",
    });
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

    const req: SleepQualityDecodeRequest = {
      embedding_id: "nonexistent-id",
      query_type: "sleep-quality",
    };

    await expect(decodeSleepQuality(req, supabase, "test-user")).rejects.toMatchObject({
      code: "EMBEDDING_NOT_FOUND",
    });
  });

  it("throws PROBE_UNAVAILABLE when ONNX adapter fails to load", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    mockLoad.mockRejectedValueOnce(new Error("ONNX file not found"));

    const req: SleepQualityDecodeRequest = {
      query_embedding: embedding,
      query_type: "sleep-quality",
    };

    await expect(decodeSleepQuality(req, supabase, "test-user")).rejects.toMatchObject({
      code: "PROBE_UNAVAILABLE",
    });
  });

  it("throws INFERENCE_FAILED when ONNX predict fails", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    mockPredict.mockRejectedValueOnce(new Error("ONNX runtime error"));

    const req: SleepQualityDecodeRequest = {
      query_embedding: embedding,
      query_type: "sleep-quality",
    };

    await expect(decodeSleepQuality(req, supabase, "test-user")).rejects.toMatchObject({
      code: "INFERENCE_FAILED",
    });
  });

  it("returns correct quality band for each score range", async () => {
    const bands: { score: number; band: string }[] = [
      { score: 0.35, band: "poor" },
      { score: 0.50, band: "fair" },
      { score: 0.75, band: "good" },
      { score: 0.90, band: "excellent" },
    ];

    for (const { score, band } of bands) {
      mockPredict.mockResolvedValue({
        values: { class_0: score },
        modelId: "sleep-quality-v1",
        durationMs: 0.3,
      });

      const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
      const supabase = createMockSupabase();
      const req: SleepQualityDecodeRequest = { query_embedding: embedding, query_type: "sleep-quality" };

      const result = await decodeSleepQuality(req, supabase, "test-user");
      expect(result.results[0].band).toBe(band);
      expect(result.results[0].score).toBeCloseTo(score, 2);
    }
  });

  it("includes provenance with all 4 artifact SHAs", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepQualityDecodeRequest = { query_embedding: embedding, query_type: "sleep-quality" };

    const result = await decodeSleepQuality(req, supabase, "test-user");

    expect(result.provenance.artifact_shas).toHaveProperty("cbramod");
    expect(result.provenance.artifact_shas).toHaveProperty("v2");
    expect(result.provenance.artifact_shas).toHaveProperty("pca");
    expect(result.provenance.artifact_shas).toHaveProperty("eegpt");
  });

  it("includes probe_sha256 in metadata", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepQualityDecodeRequest = { query_embedding: embedding, query_type: "sleep-quality" };

    const result = await decodeSleepQuality(req, supabase, "test-user");

    expect(result.metadata.probe_sha256).toBe(
      SLEEP_QUALITY_PROBE_JOINT_2312.sha256,
    );
  });

  it("returns response shape with all required fields", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: SleepQualityDecodeRequest = { query_embedding: embedding, query_type: "sleep-quality" };

    const result = await decodeSleepQuality(req, supabase, "test-user");

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
  });
});
