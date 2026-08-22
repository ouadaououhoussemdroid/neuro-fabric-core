/**
 * M34 — Unit tests for the Anomaly Detection decode service.
 *
 * Tests the core `detectAnomalies()` logic with a mock ONNX adapter
 * and mock Supabase client. Mirrors the M33 cognitive-decode test patterns.
 */
import { describe, it, expect, beforeEach, vi, type Mock } from "vitest";
import { ANOMALY_MAHALANOBIS_PROBE_JOINT_2312 } from "@/lib/ai/decoders/anomaly.registry";

// ─── Mock ONNX adapter ─────────────────────────────────────────────────

const { mockPredict, mockBuildServiceProvenance } = vi.hoisted(() => ({
  mockPredict: vi.fn(),
  mockBuildServiceProvenance: vi.fn(),
}));

vi.mock("@/lib/ai/adapters/onnx-adapter", () => {
  return {
    ONNXAdapter: class {
      readonly descriptor: { id: string };
      constructor(opts: { id: string; [key: string]: unknown }) {
        this.descriptor = { id: opts.id };
      }
      predict = mockPredict;
      load = vi.fn().mockResolvedValue(undefined);
      isLoaded = vi.fn().mockReturnValue(true);
      unload = vi.fn().mockResolvedValue(undefined);
      setModel = vi.fn();
    },
  };
});

vi.mock("@/lib/ai/services/provenance.server", () => ({
  buildServiceProvenance: (...args: unknown[]) => mockBuildServiceProvenance(...args),
}));

// ─── Now import the modules under test ──────────────────────────────────────

import {
  detectAnomalies,
  type AnomalyDetectRequest,
  AnomalyDetectError,
  ANOMALY_SERVICE,
  ANOMALY_VERSION,
  ANOMALY_DEFAULT_HEAD_ID,
  resetAnomalyProbe,
} from "../anomaly.server";
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

// ─── Tests ────────────────────────────────────────────────────────────────

describe("detectAnomalies — Anomaly Detection Service", () => {
  beforeEach(() => {
    resetAnomalyProbe();

    // Default mock behavior
    mockPredict.mockResolvedValue({
      values: { class_0: 0.85 },
      modelId: "anomaly-mahalanobis-v1",
      durationMs: 0.35,
    });

    mockBuildServiceProvenance.mockReturnValue({
      service: ANOMALY_SERVICE,
      service_version: ANOMALY_VERSION,
      embedding_model: "onnx-cbramod-joint-2312",
      embedding_dim: 2312,
      task_head_id: "anomaly-mahalanobis-v1",
      task_head_version: "0.1.0",
      task_head_sha256: "b72373576376f7c8ec2209cfe7c640033ddf13378646f01741cdd1a6c8bb9f59",
      task_head_dataset: "PhysioNet EEGMMIDB (artifact detection proxy)",
      task_head_metrics: ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.training?.metrics ?? {
        auc_roc: 0.0, f1_score: 0.0, threshold: 0, precision: 0.0, recall: 0.0,
      },
      experiment_id: "m34-anomaly-detection-probe",
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

  it("detects anomalies from a raw 2312-D query_embedding", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: AnomalyDetectRequest = { query_embedding: embedding, query_type: "artifact" };

    const result = await detectAnomalies(req, supabase, "test-user");

    expect(result.service).toBe(ANOMALY_SERVICE);
    expect(result.head).toBe(ANOMALY_DEFAULT_HEAD_ID);
    expect(result.results).toHaveLength(1);
    expect(result.results[0].metric).toBe("artifact");
    expect(result.results[0].score).toBeGreaterThanOrEqual(0);
    expect(result.results[0].score).toBeLessThanOrEqual(1);
    // Mock returns 0.85 which is above threshold (0.75)
    expect(result.results[0].is_anomalous).toBe(true);
    expect(result.results[0].confidence_interval).toHaveLength(2);
    expect(result.results[0].confidence).toBeGreaterThan(0);
  });

  it("detects anomalies from an existing embedding_id (embed-once-reuse)", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.02);
    const supabase = createMockSupabase(embedding);
    const req: AnomalyDetectRequest = {
      embedding_id: "test-embedding-id",
      query_type: "artifact",
    };

    const result = await detectAnomalies(req, supabase, "test-user");

    expect(result.metadata.embedding_reused).toBe(true);
    expect(result.embedding_id).toBe("test-embedding-id");
    expect(result.timings.embed_ms).toBeDefined();
  });

  it("throws INVALID_REQUEST when neither embedding_id nor query_embedding provided", async () => {
    const supabase = createMockSupabase();
    const req: AnomalyDetectRequest = { query_type: "artifact" };

    await expect(detectAnomalies(req, supabase, "test-user")).rejects.toThrow(
      AnomalyDetectError,
    );
  });

  it("throws DIMENSION_MISMATCH for wrong embedding dimension", async () => {
    const supabase = createMockSupabase();
    const req: AnomalyDetectRequest = {
      query_embedding: [0.1, 0.2, 0.3], // too short
      query_type: "artifact",
    };

    await expect(detectAnomalies(req, supabase, "test-user")).rejects.toThrow(
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
    const req: AnomalyDetectRequest = {
      embedding_id: "nonexistent-id",
      query_type: "artifact",
    };

    await expect(detectAnomalies(req, supabase as never, "test-user")).rejects.toThrow(
      "embedding_id not found",
    );
  });

  it("returns provenance with all 4 artifact SHAs", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.03);
    const supabase = createMockSupabase();
    const req: AnomalyDetectRequest = { query_embedding: embedding };

    const result = await detectAnomalies(req, supabase, "test-user");

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
    const req: AnomalyDetectRequest = { query_embedding: embedding };

    const result = await detectAnomalies(req, supabase, "test-user");

    expect(result.provenance.service).toBe(ANOMALY_SERVICE);
    expect(result.provenance.embedding_model).toBe("onnx-cbramod-joint-2312");
    expect(result.provenance.embedding_dim).toBe(2312);
    expect(result.provenance.task_head_id).toBe("anomaly-mahalanobis-v1");
    expect(result.provenance.task_head_metrics?.auc_roc).toBeCloseTo(0.892, 2);
    expect(result.provenance.task_head_metrics?.f1_score).toBeCloseTo(0.81, 2);
  });

  it("includes timings in response", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.05);
    const supabase = createMockSupabase();
    const req: AnomalyDetectRequest = { query_embedding: embedding };

    const result = await detectAnomalies(req, supabase, "test-user");

    expect(result.timings.inference_ms).toBeDefined();
    expect(result.timings.total_ms).toBeDefined();
    expect(typeof result.timings.inference_ms).toBe("number");
  });

  it("supports baseline query_type", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.06);
    const supabase = createMockSupabase();
    const req: AnomalyDetectRequest = {
      query_embedding: embedding,
      query_type: "baseline",
    };

    const result = await detectAnomalies(req, supabase, "test-user");
    expect(result.results[0].metric).toBe("baseline");
  });

  it("supports fatigue query_type", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.07);
    const supabase = createMockSupabase();
    const req: AnomalyDetectRequest = {
      query_embedding: embedding,
      query_type: "fatigue",
    };

    const result = await detectAnomalies(req, supabase, "test-user");
    expect(result.results[0].metric).toBe("fatigue");
  });

  it("default query_type is artifact when omitted", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.08);
    const supabase = createMockSupabase();
    const req: AnomalyDetectRequest = { query_embedding: embedding };

    const result = await detectAnomalies(req, supabase, "test-user");
    expect(result.results[0].metric).toBe("artifact");
  });

  it("anomaly score is clamped to [0, 1]", async () => {
    // Override mock to return an out-of-range prediction
    mockPredict.mockResolvedValueOnce({
      values: { class_0: 1.5 }, // out of range
      modelId: "anomaly-mahalanobis-v1",
      durationMs: 0.3,
    });

    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.09);
    const supabase = createMockSupabase();
    const req: AnomalyDetectRequest = { query_embedding: embedding };

    const result = await detectAnomalies(req, supabase, "test-user");
    expect(result.results[0].score).toBeLessThanOrEqual(1);
    expect(result.results[0].score).toBeGreaterThanOrEqual(0);
  });

  it("flags is_anomalous=true when score exceeds threshold", async () => {
    // Mock a high anomaly score (above threshold of 0.75)
    mockPredict.mockResolvedValueOnce({
      values: { class_0: 0.95 },
      modelId: "anomaly-mahalanobis-v1",
      durationMs: 0.5,
    });

    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.1);
    const supabase = createMockSupabase();
    const req: AnomalyDetectRequest = { query_embedding: embedding };

    const result = await detectAnomalies(req, supabase, "test-user");
    expect(result.results[0].score).toBeCloseTo(0.95, 2);
    expect(result.results[0].is_anomalous).toBe(true);
  });
});
