/**
 * M34 — Tests for the Anomaly Detection task-head registry.
 *
 * Tests that the anomaly detection heads are properly registered in the
 * shared TaskHeadRegistry, with correct dimensions, SHAs, and metrics.
 */
import { describe, it, expect, beforeAll } from "vitest";
import {
  registerTaskHead,
  getTaskHead,
  hasTaskHead,
  getTaskHeadsByService,
  getDefaultTaskHead,
  type TaskHeadDescriptor,
} from "../registry";
import {
  ANOMALY_HEADS,
  ANOMALY_MAHALANOBIS_PROBE_JOINT_2312,
  ANOMALY_MAHALANOBIS_PROBE_V2_32,
  registerAnomalyHeads,
  getDefaultAnomalyHead,
} from "../anomaly.registry";

describe("Anomaly TaskHeadRegistry", () => {
  beforeAll(() => {
    registerAnomalyHeads();
  });

  it("registers the anomaly-mahalanobis-v1 head with correct id", () => {
    expect(hasTaskHead(ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.id)).toBe(true);
    const head = getTaskHead("anomaly-mahalanobis-v1");
    expect(head).toBeDefined();
    expect(head?.service).toBe("anomaly-detection");
    expect(head?.inputDim).toBe(2312);
    expect(head?.outputDim).toBe(1); // anomaly score (continuous)
  });

  it("anomaly-mahalanobis-v1 uses server inference (onnxruntime-node)", () => {
    const head = getTaskHead("anomaly-mahalanobis-v1");
    expect(head?.inferenceTarget).toBe("server");
    expect(head?.sha256).toBe("b72373576376f7c8ec2209cfe7c640033ddf13378646f01741cdd1a6c8bb9f59");
  });

  it("anomaly-mahalanobis-v1 training metrics match M34 results", () => {
    const head = getTaskHead("anomaly-mahalanobis-v1");
    expect(head?.training.metrics).toBeDefined();
    expect(head?.training.metrics?.auc_roc).toBeCloseTo(0.892, 2);
    expect(head?.training.metrics?.f1_score).toBeCloseTo(0.81, 2);
    expect(head?.training.dataset).toContain("EEGMMIDB");
    expect(head?.experimentId).toBe("m34-anomaly-detection-probe");
  });

  it("registers V2-32 browser fallback head", () => {
    expect(hasTaskHead("anomaly-mahalanobis-v2-32d")).toBe(true);
    const head = getTaskHead("anomaly-mahalanobis-v2-32d");
    expect(head?.inputDim).toBe(32);
    expect(head?.outputDim).toBe(1);
    expect(head?.inferenceTarget).toBe("both");
    expect(head?.service).toBe("anomaly-detection");
  });

  it("ANOMALY_HEADS contains all registered probes", () => {
    expect(ANOMALY_HEADS.length).toBe(2);
    expect(ANOMALY_HEADS.map((h) => h.id)).toContain("anomaly-mahalanobis-v1");
    expect(ANOMALY_HEADS.map((h) => h.id)).toContain("anomaly-mahalanobis-v2-32d");
  });

  it("getTaskHeadsByService returns all anomaly heads", () => {
    const heads = getTaskHeadsByService("anomaly-detection");
    expect(heads.length).toBeGreaterThanOrEqual(2);
    expect(heads.every((h) => h.service === "anomaly-detection")).toBe(true);
  });

  it("getDefaultAnomalyHead returns server head by default", () => {
    const head = getDefaultAnomalyHead("server");
    expect(head).toBeDefined();
    expect(head?.inputDim).toBe(2312);
    expect(head?.inferenceTarget).toBe("server");
  });

  it("getDefaultAnomalyHead browser returns V2-32 head", () => {
    const head = getDefaultAnomalyHead("browser");
    expect(head).toBeDefined();
    expect(head?.inputDim).toBe(32);
    expect(head?.inferenceTarget).toBe("both");
  });

  it("all anomaly heads have correct training metadata", () => {
    for (const head of ANOMALY_HEADS) {
      expect(head.training.dataset).toBeTruthy();
      expect(head.training.protocol).toBeTruthy();
      expect(head.training.metrics).toBeDefined();
    }
  });

  it("registerAnomalyHeads is idempotent", () => {
    registerAnomalyHeads();
    registerAnomalyHeads();
    // Should still have the same head
    const head = getTaskHead("anomaly-mahalanobis-v1");
    expect(head).toBeDefined();
    expect(head?.version).toBe("0.1.0");
  });
});
