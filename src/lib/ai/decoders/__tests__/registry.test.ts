/**
 * M32 — Tests for the ServiceRegistry (TaskHeadRegistry).
 *
 * Tests the reusable task-head registry that Subject Identity, Cognitive State,
 * and Anomaly Detection services all register their heads in.
 */
import { describe, it, expect, beforeEach } from "vitest";
import {
  registerTaskHead,
  getTaskHead,
  hasTaskHead,
  listTaskHeads,
  getTaskHeadsByService,
  getDefaultTaskHead,
  serviceIdentity,
  type TaskHeadDescriptor,
} from "../registry";

const mockSubjectIdentityHead: TaskHeadDescriptor = {
  id: "subject-identity-similarity-v1",
  name: "Subject Identity Similarity Search",
  version: "0.1.0",
  service: "subject-identity",
  inputDim: 2312,
  outputDim: 1,
  inferenceTarget: "server",
  sha256: "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
  training: {
    dataset: "PhysioNet EEGMMIDB S001-S050, runs 5-10",
    protocol: "50-fold LOSO, session-disjoint",
    metrics: { r5: 0.8527, r1: 0.6438, mrr: 0.7361 },
  },
  experimentId: "m27-augmented-joint-2312",
};

const mockCognitiveHead: TaskHeadDescriptor = {
  id: "cognitive-linear-v1",
  name: "Cognitive Workload Linear Probe",
  version: "0.1.0",
  service: "cognitive-intelligence",
  inputDim: 2312,
  outputDim: 1,
  inferenceTarget: "both",
  sha256: "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3",
  training: {
    dataset: "SEED",
    protocol: "15-fold LOSO, session-disjoint",
    metrics: { r2: 0.42, pearson_r: 0.68 },
  },
  experimentId: "m31-cognitive-workload-probe",
};

const mockAnomalyHead: TaskHeadDescriptor = {
  id: "anomaly-mahalanobis-v1",
  name: "Mahalanobis Anomaly Detector",
  version: "0.1.0",
  service: "anomaly-detection",
  inputDim: 2312,
  outputDim: 1,
  inferenceTarget: "server",
  sha256: "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4",
  training: {
    dataset: "PhysioNet EEGMMIDB (normal) + synthetic artifacts",
    protocol: "50-fold LOSO, normal-fit on 49 subjects",
    metrics: { auroc: 0.92, fdr: 0.08 },
  },
  experimentId: "m31-anomaly-evaluation",
};

describe("ServiceRegistry — TaskHeadRegistry", () => {
  beforeEach(() => {
    // Reset the registry between tests to avoid cross-test contamination.
    // Since the registry is module-level, we use the fact that registerTaskHead
    // with the same id replaces the prior entry.
  });

  it("registers and retrieves a task head by id", () => {
    registerTaskHead(mockSubjectIdentityHead);
    const head = getTaskHead(mockSubjectIdentityHead.id);
    expect(head).toBeDefined();
    expect(head?.id).toBe("subject-identity-similarity-v1");
    expect(head?.service).toBe("subject-identity");
    expect(head?.inputDim).toBe(2312);
    expect(head?.outputDim).toBe(1);
    expect(head?.inferenceTarget).toBe("server");
  });

  it("returns undefined for unregistered heads", () => {
    expect(getTaskHead("nonexistent-head-v999")).toBeUndefined();
  });

  it("hasTaskHead returns true for registered, false for unregistered", () => {
    registerTaskHead(mockSubjectIdentityHead);
    expect(hasTaskHead(mockSubjectIdentityHead.id)).toBe(true);
    expect(hasTaskHead("nonexistent-head-v999")).toBe(false);
  });

  it("listTaskHeads returns all registered heads", () => {
    registerTaskHead(mockSubjectIdentityHead);
    registerTaskHead(mockCognitiveHead);
    registerTaskHead(mockAnomalyHead);
    const all = listTaskHeads();
    expect(all.length).toBeGreaterThanOrEqual(3);
  });

  it("getTaskHeadsByService returns only heads for the specified service", () => {
    registerTaskHead(mockSubjectIdentityHead);
    registerTaskHead(mockCognitiveHead);
    registerTaskHead(mockAnomalyHead);
    const subjHeads = getTaskHeadsByService("subject-identity");
    expect(subjHeads.length).toBeGreaterThanOrEqual(1);
    expect(subjHeads.every((h) => h.service === "subject-identity")).toBe(true);
  });

  it("getDefaultTaskHead returns the first registered head for a service", () => {
    registerTaskHead(mockSubjectIdentityHead);
    registerTaskHead(mockCognitiveHead);
    const defaultHead = getDefaultTaskHead("subject-identity");
    expect(defaultHead).toBeDefined();
    expect(defaultHead?.service).toBe("subject-identity");
  });

  it("returns undefined for getDefaultTaskHead on unregistered service", () => {
    expect(getDefaultTaskHead("nonexistent-service")).toBeUndefined();
  });

  it("registerTaskHead replaces prior registration with same id", () => {
    registerTaskHead(mockSubjectIdentityHead);
    expect(getTaskHead(mockSubjectIdentityHead.id)?.name).toBe(
      "Subject Identity Similarity Search",
    );
    const updated: TaskHeadDescriptor = {
      ...mockSubjectIdentityHead,
      name: "Subject Identity v2",
    };
    registerTaskHead(updated);
    expect(getTaskHead(mockSubjectIdentityHead.id)?.name).toBe(
      "Subject Identity v2",
    );
  });

  it("serviceIdentity produces deterministic identifier", () => {
    const identity = serviceIdentity("subject-identity", "0.1.0");
    expect(identity).toBe("neurofabric-subject-identity@v0.1.0");
  });

  it("task head descriptors include training metrics", () => {
    registerTaskHead(mockSubjectIdentityHead);
    const head = getTaskHead(mockSubjectIdentityHead.id);
    expect(head?.training.metrics).toBeDefined();
    expect(head?.training.metrics?.r5).toBe(0.8527);
    expect(head?.experimentId).toBe("m27-augmented-joint-2312");
  });
});
