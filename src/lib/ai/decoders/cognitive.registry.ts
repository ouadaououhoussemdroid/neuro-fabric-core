/**
 * M33 — Cognitive task-head registry.
 *
 * Registers the cognitive workload heads for the Joint-2312 (2312-D) embedding space.
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
 * Linear probe: Joint-2312 (2312-D) → cognitive workload (1-D).
 *
 * Trained as a Ridge regression head on Joint-2312 embeddings from
 * PhysioNet EEGMMIDB (S001-S050, 50 subjects). Workload proxy derived from
 * the band-power ratio heuristic (same as M31 §7.5).
 *
 * R² = 0.7348, RMSE = 0.0557, MAE = 0.0440, Pearson r = 0.8874.
 *
 * Inference: server-only (onnxruntime-node, 2312-D input is too large for
 * browser WASM inference in real-time).
 */
export const COGNITIVE_LINEAR_PROBE_JOINT_2312: TaskHeadDescriptor = {
  id: "cognitive-linear-v1",
  name: "Cognitive Workload Linear Probe (Joint-2312)",
  version: "0.1.0",
  service: "cognitive-intelligence",
  inputDim: JOINT_2312_EMBEDDING_DIM,
  outputDim: 1, // workload score [0, 1]
  inferenceTarget: "server",
  sha256: "ab8bc6389d98a9461fc7f0f4fea47c3cd9860595c305879351ad0cf6592a6b32",
  artifactUri: "/models/cognitive/cognitive-probe-joint2312-v1.onnx",
  training: {
    dataset: "PhysioNet EEGMMIDB (S001-S050, workload proxy)",
    protocol: "50-fold LOSO cross-validation, train-only Ridge regression",
    metrics: {
      r2: 0.7348,
      rmse: 0.0557,
      mae: 0.0440,
      pearson_r: 0.8874,
    },
  },
  validation: {
    loso_folds: 50,
    p_value_vs_baseline: 5.58e-28,
    baseline_r2: 0.25,
  },
  experimentId: "m33-cognitive-workload-probe",
};

/**
 * Browser fallback probe: V2-32 (32-D) → cognitive workload (1-D).
 *
 * A lightweight linear projection of the 2312-D probe, suitable for
 * browser-side inference. Achieves R²~0.35 from the V2-32 subspace.
 *
 * Inference: both (browser WASM + server).
 */
export const COGNITIVE_LINEAR_PROBE_V2_32: TaskHeadDescriptor = {
  id: "cognitive-linear-v2-32d",
  name: "Cognitive Workload Probe (V2-32 Browser Fallback)",
  version: "0.1.0",
  service: "cognitive-intelligence",
  inputDim: V2_EMBEDDING_DIM,
  outputDim: 1,
  inferenceTarget: "both",
  sha256: "3ebd9ef943b17d23f65a45cda7f73301a623e95fffda2659c9c1245444e720c1",
  artifactUri: "/models/cognitive/cognitive-probe-v2-32d-v1.onnx",
  training: {
    dataset: "PhysioNet EEGMMIDB (S001-S050, workload proxy, V2-32 projected)",
    protocol: "50-fold LOSO cross-validation on V2-32 embeddings",
    metrics: {
      r2: 0.35,
      rmse: 0.12,
    },
  },
  validation: {
    loso_folds: 50,
    note: "Browser fallback — trained on V2-32 embeddings via Ridge regression",
  },
  experimentId: "m33-cognitive-workload-probe",
};

/**
 * MLP probe: Joint-2312 (2312-D) → cognitive workload (1-D).
 *
 * A non-linear MLP fallback (3-layer: 2312→128→64→1) that is used only if
 * the linear probe R² < 0.40. The linear probe achieves R²=0.7348, well
 * above the 0.40 threshold, so the MLP is not active in the production path.
 *
 * Not yet trained — SHA set to empty string (no artifact on disk).
 * Will be populated with a real SHA when the MLP fallback head is trained.
 *
 * Inference: server-only.
 */
export const COGNITIVE_MLP_PROBE_JOINT_2312: TaskHeadDescriptor = {
  id: "cognitive-mlp-v1",
  name: "Cognitive Workload MLP Probe (Joint-2312)",
  version: "0.1.0",
  service: "cognitive-intelligence",
  inputDim: JOINT_2312_EMBEDDING_DIM,
  outputDim: 1,
  inferenceTarget: "server",
  sha256: "",
  artifactUri: "/models/cognitive/cognitive-mlp-probe-joint2312-v1.onnx",
  training: {
    dataset: "Not yet trained",
    protocol: "Conditional fallback only (activated if linear probe R² < 0.40)",
    metrics: {},
  },
  validation: {
    note: "Not yet trained — used only if linear probe R² < 0.40.",
  },
  experimentId: "m33-cognitive-workload-probe",
};

/** All cognitive task heads, in priority order. */
export const COGNITIVE_HEADS: TaskHeadDescriptor[] = [
  COGNITIVE_LINEAR_PROBE_JOINT_2312,
  COGNITIVE_LINEAR_PROBE_V2_32,
  COGNITIVE_MLP_PROBE_JOINT_2312,
];

/**
 * Register all cognitive task heads into the shared TaskHeadRegistry.
 * Call this at application startup (server) or module load (browser).
 */
export function registerCognitiveHeads(): void {
  for (const head of COGNITIVE_HEADS) {
    registerTaskHead(head);
  }
}

/** Convenience: get the default cognitive head for a given inference target. */
export function getDefaultCognitiveHead(
  target: "server" | "browser" | "both" = "server",
): TaskHeadDescriptor | undefined {
  const heads = COGNITIVE_HEADS.filter(
    (h) => h.inferenceTarget === target || h.inferenceTarget === "both",
  );
  return heads[0];
}
