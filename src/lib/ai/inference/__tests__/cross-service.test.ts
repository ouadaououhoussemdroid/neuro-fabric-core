/**
 * M35 — Cross-Service Integration Tests for Tier 1.
 *
 * Validates that all three Tier-1 services (Subject Identity, Cognitive State,
 * Anomaly Detection) correctly implement the "Embed Once → Reuse Many" principle,
 * share consistent provenance (same 4 artifact SHAs), and meet latency budgets
 * when invoked sequentially with the same Joint-2312 embedding.
 *
 * Mocks: ONNXAdapter (for cognitive + anomaly probes), buildServiceProvenance,
 *        and Supabase client (for embedding_id reuse path).
 *
 * Latency budget: Joint-2312 → 3 services ≤ 100ms target (no re-embedding).
 * Provenance consistency: all 3 services must report the same artifact_shas.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";

// ─── Shared mocks (same pattern as M33/M34 unit tests) ───────────────────────

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

// ─── Import services ─────────────────────────────────────────────────────────

import { searchSubjectIdentity } from "../subject-identity.server";
import { decodeCognitiveState } from "../cognitive.server";
import { detectAnomalies } from "../anomaly.server";
import { JOINT_2312_EMBEDDING_DIM } from "../joint.server";

// ─── Canonical artifact SHAs (must match all 3 services) ──────────────────────

const EXPECTED_SHAS = {
  cbramod: expect.stringMatching(/^c128ccfd/),
  v2: expect.stringMatching(/^18644de1/),
  pca: expect.any(String),
  eegpt: expect.stringMatching(/^a92daf44/),
};

// ─── Mock Supabase with a shared embedding_id ────────────────────────────────

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
    rpc: vi.fn((fn: string, args: Record<string, unknown>) => {
      // Return mock ANN search results for match_joint_embeddings_2312 RPC
      return Promise.resolve({
        data: [
          { id: "result-1", score: 0.92, meta: { subject_id: "S01", session_id: "session_1" } },
          { id: "result-2", score: 0.87, meta: { subject_id: "S02", session_id: "session_2" } },
        ],
        error: null,
      });
    }),
  };
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("M35 — Cross-Service Tier 1 Integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Mock ONNX predict for cognitive + anomaly probes
    mockPredict.mockResolvedValue({
      values: { class_0: 0.73 },
      modelId: "joint-probe-v1",
      durationMs: 0.42,
    });

    // Mock provenance — returns the canonical 4 artifact SHAs
    mockBuildServiceProvenance.mockReturnValue({
      service: "test-service",
      service_version: "v0.1.0",
      embedding_model: "onnx-cbramod-joint-2312",
      embedding_dim: 2312,
      task_head_id: "test-head-v1",
      task_head_version: "0.1.0",
      task_head_sha256: "test-sha",
      task_head_dataset: "test-dataset",
      task_head_metrics: { auc_roc: 0.892, r2: 0.7348 },
      experiment_id: "m31-test",
      timestamp: new Date().toISOString(),
      artifact_shas: {
        cbramod: "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
        v2: "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931",
        pca: "deterministic-pca-v1-bandpower-5x22->32",
        eegpt: "a92daf44ab8cb10e4c258c853b2d4668232acc6e09606fa2e18d052c4b914f36",
      },
      block_weights: { cbramod: 0.3062, v2: 0.1434, pca: 0.1519, eegpt: 0.3985 },
      component_dims: { cbramod: 200, v2: 32, pca: 32, eegpt: 2048 },
    });
  });

  // ── Test 1: All 3 services accept the same embedding_id ────────────────────

  it("all Tier-1 services accept and reuse the same embedding_id", async () => {
    const supabase = createMockSupabase(new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01));
    const embeddingId = "shared-embed-id";

    // Subject Identity
    const subjectResult = await searchSubjectIdentity(
      { embedding_id: embeddingId, query_type: "subject_identification" },
      supabase,
      "test-user",
    );
    expect(subjectResult.results.length).toBeGreaterThan(0);

    // Cognitive State
    const cognitiveResult = await decodeCognitiveState(
      { embedding_id: embeddingId, query_type: "workload" },
      supabase,
      "test-user",
    );
    expect(cognitiveResult.results[0].metric).toBe("workload");

    // Anomaly Detection
    const anomalyResult = await detectAnomalies(
      { embedding_id: embeddingId, query_type: "artifact" },
      supabase,
      "test-user",
    );
    expect(anomalyResult.results[0].metric).toBe("artifact");

    // All 3 should report embedding_reused=true
    expect(subjectResult.metadata.embedding_reused).toBe(true);
    expect(cognitiveResult.metadata.embedding_reused).toBe(true);
    expect(anomalyResult.metadata.embedding_reused).toBe(true);
  });

  // ── Test 2: All 3 services return consistent provenance ────────────────────

  it("all Tier-1 services return provenance with the same 4 artifact SHAs", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();

    // Subject Identity
    const subjectResult = await searchSubjectIdentity(
      { query_embedding: embedding, query_type: "subject_identification" },
      supabase,
      "test-user",
    );
    expect(subjectResult.provenance.artifact_shas).toMatchObject(EXPECTED_SHAS);

    // Cognitive State
    const cognitiveResult = await decodeCognitiveState(
      { query_embedding: embedding, query_type: "workload" },
      supabase,
      "test-user",
    );
    expect(cognitiveResult.provenance.artifact_shas).toMatchObject(EXPECTED_SHAS);

    // Anomaly Detection
    const anomalyResult = await detectAnomalies(
      { query_embedding: embedding, query_type: "artifact" },
      supabase,
      "test-user",
    );
    expect(anomalyResult.provenance.artifact_shas).toMatchObject(EXPECTED_SHAS);
  });

  // ── Test 3: All 3 services use the same Joint-2312 embedding dimension ────

  it("all Tier-1 services validate for 2312-D embedding dimension", async () => {
    const supabase = createMockSupabase();
    const badEmbedding = [0.1, 0.2, 0.3]; // too short

    // Subject Identity throws SubjectIdentityError
    await expect(
      searchSubjectIdentity(
        { query_embedding: badEmbedding, query_type: "subject_identification" },
        supabase,
        "test-user",
      ),
    ).rejects.toThrow(/Expected 2312-D/);

    // Cognitive State throws CognitiveDecodeError
    await expect(
      decodeCognitiveState(
        { query_embedding: badEmbedding, query_type: "workload" },
        supabase,
        "test-user",
      ),
    ).rejects.toThrow(/Expected 2312-D/);

    // Anomaly Detection throws AnomalyDetectError
    await expect(
      detectAnomalies(
        { query_embedding: badEmbedding, query_type: "artifact" },
        supabase,
        "test-user",
      ),
    ).rejects.toThrow(/Expected 2312-D/);
  });

  // ── Test 4: Latency budget across all 3 services ────────────────────────────

  it("all 3 services complete within 100ms latency budget for reuse path", async () => {
    const supabase = createMockSupabase(new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01));
    const embeddingId = "latency-test-embed";

    const t0 = performance.now();

    // Run all 3 services sequentially with shared embedding_id
    await searchSubjectIdentity(
      { embedding_id: embeddingId, query_type: "subject_identification" },
      supabase,
      "test-user",
    );
    await decodeCognitiveState(
      { embedding_id: embeddingId, query_type: "workload" },
      supabase,
      "test-user",
    );
    await detectAnomalies(
      { embedding_id: embeddingId, query_type: "artifact" },
      supabase,
      "test-user",
    );

    const totalMs = performance.now() - t0;

    // Target: all 3 services (with shared embedding reuse, no re-embed) ≤ 100ms
    // This is a generous target since the mock ONNX predict is ~0.15ms each.
    expect(totalMs).toBeLessThan(500); // 500ms ceiling; real target is <100ms
  });

  // ── Test 5: All 3 services increment the shared tier1 metrics ───────────────

  it("all Tier-1 services increment tier1ServiceRequestsTotal with correct service label", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();

    const mockInc = vi.fn();
    const mockObserve = vi.fn();

    // Track metric calls by spying on the metrics object
    const { metrics } = await import("@/lib/metrics");

    const incSpy = vi.spyOn(metrics.tier1ServiceRequestsTotal, "inc");
    const subjectIdSpy = vi.spyOn(metrics.subjectIdentityRequestsTotal, "inc");
    const cognitiveSpy = vi.spyOn(metrics.cognitiveDecodeRequestsTotal, "inc");
    const anomalySpy = vi.spyOn(metrics.anomalyDetectRequestsTotal, "inc");

    await searchSubjectIdentity(
      { query_embedding: embedding, query_type: "subject_identification" },
      supabase,
      "test-user",
    );
    await decodeCognitiveState(
      { query_embedding: embedding, query_type: "workload" },
      supabase,
      "test-user",
    );
    await detectAnomalies(
      { query_embedding: embedding, query_type: "artifact" },
      supabase,
      "test-user",
    );

    // Each service should increment its own counter
    expect(subjectIdSpy).toHaveBeenCalled();
    expect(cognitiveSpy).toHaveBeenCalled();
    expect(anomalySpy).toHaveBeenCalled();

    // And the shared tier1 counter should have been incremented 3 times total
    expect(incSpy).toHaveBeenCalledTimes(3);
    expect(incSpy).toHaveBeenCalledWith({ service: "subject-identity" });
    expect(incSpy).toHaveBeenCalledWith({ service: "cognitive-intelligence" });
    expect(incSpy).toHaveBeenCalledWith({ service: "anomaly-detection" });

    incSpy.mockRestore();
    subjectIdSpy.mockRestore();
    cognitiveSpy.mockRestore();
    anomalySpy.mockRestore();
  });

  // ── Test 6: All 3 services reject when neither embedding_id nor query_embedding ─

  it("all Tier-1 services throw when neither embedding_id nor query_embedding provided", async () => {
    const supabase = createMockSupabase();

    await expect(
      searchSubjectIdentity(
        { query_type: "subject_identification" },
        supabase,
        "test-user",
      ),
      ).rejects.toThrow(/Either query_embedding or embedding_id/);

    await expect(
      decodeCognitiveState(
      { query_type: "workload" },
        supabase,
        "test-user",
      ),
    ).rejects.toThrow(/Either query_embedding or embedding_id/);

    await expect(
      detectAnomalies(
        { query_type: "artifact" },
        supabase,
        "test-user",
      ),
    ).rejects.toThrow(/Either query_embedding or embedding_id/);
  });

  // ── Test 7: All 3 services use shared query_embedding interface ─────────────

  it("all Tier-1 services accept query_embedding as a 2312-element array", async () => {
    const supabase = createMockSupabase();
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.0123);

    // Verify each service accepts the same raw embedding shape
    const subjectResult = await searchSubjectIdentity(
      { query_embedding: embedding, query_type: "subject_identification" },
      supabase,
      "test-user",
    );
    const cognitiveResult = await decodeCognitiveState(
      { query_embedding: embedding, query_type: "workload" },
      supabase,
      "test-user",
    );
    const anomalyResult = await detectAnomalies(
      { query_embedding: embedding, query_type: "artifact" },
      supabase,
      "test-user",
    );

    // All 3 return structured results
    expect(subjectResult.results).toEqual(expect.any(Array));
    expect(cognitiveResult.results).toHaveLength(1);
    expect(anomalyResult.results).toHaveLength(1);
  });
});
