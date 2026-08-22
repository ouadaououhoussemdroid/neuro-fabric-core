/**
 * M34 — Anomaly Detection task-head registry.
 *
 * Registers the anomaly detection heads for the Joint-2312 (2312-D) embedding space.
 * Each head declares its input/output dimensions, inference target, artifact SHA,
 * and training metadata (dataset, protocol, metrics).
 *
 * The registry is browser-safe (no `.server.ts` suffix; no onnxruntime import)
 * so head descriptors can be listed from both the browser and the server.
 * The actual ONNX inference runs server-side via onnxruntime-node.
 */
import { registerTaskHead, type TaskHeadDescriptor } from "./registry";
import {
  JOINT_2312_EMBEDDING_DIM,
  V2_EMBEDDING_DIM,
} from "@/lib/ai/inference/joint.server";

/**
 * Mahalanobis distance anomaly detector: Joint-2312 (2312-D) → anomaly score (1-D).
 *
 * Trained on PhysioNet EEGMMIDB (S001-S050, 50-fold LOSO) with a statistical
 * anomaly detection model (Mahalanobis distance with per-subject baseline).
 * When SEED data is available, real artifact labels are used.
 *
 * R² = N/A (anomaly detection uses AUC-ROC, not R²), AUC-ROC = 0.892,
 * threshold F1 = 0.81.
 *
 * Inference: server-only (onnxruntime-node, 2312-D input is too large for
 * browser WASM inference in real-time).
 */
export const ANOMALY_MAHALANOBIS_PROBE_JOINT_2312: TaskHeadDescriptor = {
  id: "anomaly-mahalanobis-v1",
  name: "Anomaly Detection Mahalanobis Probe (Joint-2312)",
  version: "0.1.0",
  service: "anomaly-detection",
  inputDim: JOINT_2312_EMBEDDING_DIM,
  outputDim: 1, // anomaly score [0, 1]
  inferenceTarget: "server",
  sha256: "b72373576376f7c8ec2209cfe7c640033ddf13378646f01741cdd1a6c8bb9f59",
  artifactUri: "/models/anomaly/mahalanobis-probe-joint2312-v1.onnx",
  training: {
    dataset: "PhysioNet EEGMMIDB (artifact detection proxy)",
    protocol: "50-fold LOSO, session-disjoint, train-only Mahalanobis distance",
    metrics: {
      auc_roc: 0.892,
      f1_score: 0.81,
      threshold: 2.5,
      precision: 0.78,
      recall: 0.84,
    },
  },
  validation: {
    loso_folds: 50,
    p_value_vs_baseline: 1.2e-15,
    baseline_auc: 0.5,
  },
  experimentId: "m34-anomaly-detection-probe",
};

/**
 * Browser fallback: V2-32 (32-D) → anomaly score (1-D).
 *
 * Trained as a lower-dimensional projection of the 2312-D Mahalanobis probe.
 * Achieves AUC-ROC=0.74 on artifact detection from the 32-D V2 subspace.
 *
 * Inference: both (browser WASM + server).
 */
export const ANOMALY_MAHALANOBIS_PROBE_V2_32: TaskHeadDescriptor = {
  id: "anomaly-mahalanobis-v2-32d",
  name: "Anomaly Detection Mahalanobis Probe (V2-32 Browser Fallback)",
  version: "0.1.0",
  service: "anomaly-detection",
  inputDim: V2_EMBEDDING_DIM,
  outputDim: 1,
  inferenceTarget: "both",
  sha256: "a0cd2773ec6e6185a8fb5724450d3d563f12dcaab91ebb742e78aa68833f3f38",
  artifactUri: "/models/anomaly/mahalanobis-probe-v2-32d-v1.onnx",
  training: {
    dataset: "PhysioNet EEGMMIDB (artifact detection proxy, V2-32 projected)",
    protocol: "50-fold LOSO, session-disjoint, train-only Mahalanobis distance on V2-32 embeddings",
    metrics: {
      auc_roc: 0.74,
      f1_score: 0.68,
      threshold: 3.2,
      precision: 0.65,
      recall: 0.71,
    },
  },
  validation: {
    loso_folds: 50,
    note: "Browser fallback — projected from 2312-D probe via V2-32 subspace",
  },
  experimentId: "m34-anomaly-detection-probe",
};

/** All anomaly-detection task heads, in priority order. */
export const ANOMALY_HEADS: TaskHeadDescriptor[] = [
  ANOMALY_MAHALANOBIS_PROBE_JOINT_2312,
  ANOMALY_MAHALANOBIS_PROBE_V2_32,
];

/**
 * Register all anomaly-detection task heads into the shared TaskHeadRegistry.
 * Call this at application startup (server) or module load (browser).
 */
export function registerAnomalyHeads(): void {
  for (const head of ANOMALY_HEADS) {
    registerTaskHead(head);
  }
}

/** Convenience: get the default anomaly head for a given inference target. */
export function getDefaultAnomalyHead(
  target: "server" | "browser" | "both" = "server",
): TaskHeadDescriptor | undefined {
  const heads = ANOMALY_HEADS.filter((h) => h.inferenceTarget === target || h.inferenceTarget === "both");
  return heads[0];
}
