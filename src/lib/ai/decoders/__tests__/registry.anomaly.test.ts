/**
 * M34-Scientific-Reboot — Tests for the Anomaly Detection task-head registry.
 *
 * Tests that the anomaly detection heads are properly registered in the
 * shared TaskHeadRegistry, with correct dimensions, SHAs, scientific status,
 * and methodology consistency (ONNX = CV).
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

  it("registers the anomaly-mahalanobis-v2 head with correct id", () => {
    expect(hasTaskHead(ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.id)).toBe(true);
    const head = getTaskHead("anomaly-mahalanobis-v2");
    expect(head).toBeDefined();
    expect(head?.service).toBe("anomaly-detection");
    expect(head?.inputDim).toBe(2312);
    expect(head?.outputDim).toBe(1);
  });

  it("anomaly-mahalanobis-v2 uses server inference", () => {
    const head = getTaskHead("anomaly-mahalanobis-v2");
    expect(head?.inferenceTarget).toBe("server");
    expect(head?.sha256).toBe("b72373576376f7c8ec2209cfe7c640033ddf13378646f01741cdd1a6c8bb9f59");
  });

  it("anomaly-mahalanobis-v2 training reflects V2 (not stale V1)", () => {
    const head = getTaskHead("anomaly-mahalanobis-v2");
    expect(head?.training.metrics).toBeDefined();
    expect(head?.training.metrics?.auc_roc).toBeCloseTo(0.4757, 2);
    expect(head?.training.dataset).not.toContain("proxy");
    expect(head?.experimentId).toBe("m34-anomaly-detection-probe-v2");
  });

  it("anomaly-mahalanobis-v2 methodology match verified (Mahalanobis CV = ONNX)", () => {
    const head = getTaskHead("anomaly-mahalanobis-v2");
    expect(head?.training.protocol).toContain("Mahalanobis");
    expect(head?.scientificStatus).toBe("EXPERIMENTAL");
    expect(head?.previousMetrics?.status).toBe("INVALID");
    expect(head?.previousMetrics?.reason).toContain("Ridge regression");
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
    expect(ANOMALY_HEADS.map((h) => h.id)).toContain("anomaly-mahalanobis-v2");
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

  it("all anomaly heads have scientific classification", () => {
    for (const head of ANOMALY_HEADS) {
      expect(head.scientificStatus).toBeDefined();
      expect(head.scientificStatus).toMatch(
        /SCIENTIFICALLY_VALIDATED|ENGINEERING_VALIDATED|EXPERIMENTAL|PROXY_DEMONSTRATION|BLOCKED/,
      );
    }
  });

  it("registerAnomalyHeads is idempotent", () => {
    registerAnomalyHeads();
    registerAnomalyHeads();
    const head = getTaskHead("anomaly-mahalanobis-v2");
    expect(head).toBeDefined();
    expect(head?.version).toBe("0.2.0");
  });
});
