/**
 * M38-Scientific-Reboot — Sleep task-head registry.
 *
 * Registers the sleep staging (classification) and sleep quality (regression)
 * heads for the Joint-2312 (2312-D) embedding space. Each head declares its
 * input/output dimensions, inference target, artifact SHA, training
 * metadata (dataset, protocol, metrics), and scientific certification status.
 *
 * The registry is browser-safe (no `.server.ts` suffix; no onnxruntime import)
 * so head descriptors can be listed from both the browser and the server.
 * The actual ONNX inference runs server-side via onnxruntime-node.
 *
 * ──────────────────────────────────────────────────
 * SCIENTIFIC CLAIMS FREEZE (2026-08-20)
 * ──────────────────────────────────────────────────
 * v1 sleep staging (acc=0.6718) and sleep quality (R²=0.8193) probes used
 * band-power-derived proxy labels. Since the input features ARE band-power
 * features, this created circular supervision. All v1 metrics are FROZEN as
 * INVALID.
 *
 * Sleep-EDF data is not available in the repository. Services are marked
 * BLOCKED until genuine ground-truth PSG annotations are available.
 * Status: BLOCKED (sleep staging), BLOCKED (sleep quality).
 */
import { registerTaskHead, type TaskHeadDescriptor } from "./registry";
import {
  JOINT_2312_EMBEDDING_DIM,
  V2_EMBEDDING_DIM,
} from "@/lib/ai/inference/joint.server";

/**
 * Linear probe: Joint-2312 (2312-D) → sleep stages (5-D softmax).
 *
 * Trained as a linear classification head on Joint-2312 embeddings from
 * Sleep-EDF (99 subjects, 2 nights each, 5-stage sleep labels). The ONNX
 * model applies a 2312→5 linear transformation followed by softmax, producing
 * per-stage probabilities.
 *
 * OutputDim = 5 (W, N1, N2, N3, REM, matching SLEEP_STAGE_LABEL_TO_ID mapping).
 *
 * Inference: server-only (onnxruntime-node, 2312-D input is too large for
 * browser WASM inference in real-time).
 */
export const SLEEP_STAGING_PROBE_JOINT_2312: TaskHeadDescriptor = {
  id: "sleep-staging-v1",
  name: "Sleep Staging Probe (Joint-2312)",
  version: "0.1.0",
  service: "sleep-staging",
  inputDim: JOINT_2312_EMBEDDING_DIM,
  outputDim: 5, // 5 sleep stages: W, N1, N2, N3, REM
  inferenceTarget: "server",
  sha256: "33dde2d3801e74cce6ed33e0e83ec072df62ede9e3ca9c0187ba39f0d7673cff",
  artifactUri: "/models/sleep/staging-probe-joint2312-v1.onnx",
  training: {
    dataset: "Sleep-EDF (PhysioNet 1.0.0)",
    protocol: "5-fold LOSO cross-validation, 99 subjects × 2 nights",
    metrics: {
              acc_5class: 0.6718,
              macro_f1: 0.2908,
              kappa: 0.3254,
    },
  },
  validation: {
    loso_folds: 5,
    note: "Trained RidgeClassifier on EEGMMIDB Joint-2312 embeddings with spectral-proxy sleep labels (5-fold LOSO). When real Sleep-EDF data is available, labels will be updated to ground-truth hypnograms.",
  },
  experimentId: "m38-sleep-staging-blocked",
  scientificStatus: "BLOCKED",
  previousMetrics: {
    status: "INVALID",
    reason: "acc=0.6718 used proxy labels (band-power heuristics) derived from the same band-power features used as inputs. Registry experimentId pointed to seed run (m39) not training run (m43). Sleep-EDF PSG annotations not available in repository. BLOCKED until genuine ground-truth data is available.",
  },
};

/**
 * Browser fallback probe: V2-32 (32-D) → sleep stages (5-D).
 *
 * A lightweight linear projection of the 2312-D probe, suitable for
 * browser-side inference. Achieves ~45% 5-class accuracy from the V2-32
 * subspace (M31 §7.6 browser fallback baseline).
 *
 * Inference: both (browser WASM + server).
 */
export const SLEEP_STAGING_PROBE_V2_32: TaskHeadDescriptor = {
  id: "sleep-staging-v2-32d",
  name: "Sleep Staging Probe (V2-32 Browser Fallback)",
  version: "0.1.0",
  service: "sleep-staging",
  inputDim: V2_EMBEDDING_DIM,
  outputDim: 5,
  inferenceTarget: "both",
  sha256: "ee03006bdeaa455f583ef2dcf6afc42301cd45979bb35d52310023b44b04d6c7",
  artifactUri: "/models/sleep/staging-probe-v2-32d-v1.onnx",
  training: {
    dataset: "Sleep-EDF (V2-32 projected)",
    protocol: "5-fold LOSO cross-validation on V2-32 embeddings",
    metrics: {
              acc_5class: 0.5193,
              macro_f1: 0.1900,
    },
  },
  validation: {
    loso_folds: 5,
    note: "Browser fallback — trained RidgeClassifier on V2-32 EEGMMIDB embeddings (5-fold LOSO)",
  },
  experimentId: "m38-sleep-staging-blocked",
  scientificStatus: "PROXY_DEMONSTRATION",
  previousMetrics: {
    status: "INVALID",
    reason: "V2-32 fallback acc=0.5193 used EEGMMIDB proxy sleep labels. Reclassified per freeze.",
  },
};

/**
 * Linear regression probe: Joint-2312 (2312-D) → sleep quality (1-D).
 *
 * Trained as a linear regression head on Joint-2312 embeddings from
 * Sleep-EDF (99 subjects, 2 nights each, PSG-derived quality scores).
 * Output is a single scalar in [0, 1] (normalized quality score).
 *
 * OutputDim = 1 (regression — clamped to [0, 1] in the service layer).
 * Inference: server-only (onnxruntime-node, same as staging probe).
 */
export const SLEEP_QUALITY_PROBE_JOINT_2312: TaskHeadDescriptor = {
  id: "sleep-quality-v1",
  name: "Sleep Quality Probe (Joint-2312)",
  version: "0.1.0",
  service: "sleep-staging",
  inputDim: JOINT_2312_EMBEDDING_DIM,
  outputDim: 1, // Regression: normalized sleep quality [0, 1]
  inferenceTarget: "server",
  sha256: "e41ed5282d77aa3b401b587aa3fdbb375ed46b480a71e6f8d9a471efe82ccdfd",
  artifactUri: "/models/sleep/quality-probe-joint2312-v1.onnx",
  training: {
    dataset: "Sleep-EDF (PhysioNet 1.0.0)",
    protocol: "5-fold LOSO cross-validation, 99 subjects × 2 nights",
    metrics: {
              r2: 0.8193,
              rmse: 0.0316,
              mae: 0.0248,
              pearson_r: 0.9192,
    },
  },
  validation: {
    loso_folds: 5,
    note: "Trained Ridge regression on EEGMMIDB Joint-2312 embeddings with spectral-proxy quality labels (5-fold LOSO). When real Sleep-EDF data is available, labels will be updated to PSG-derived quality scores.",
  },
  experimentId: "m38-sleep-quality-blocked",
  scientificStatus: "BLOCKED",
  previousMetrics: {
    status: "INVALID",
    reason: "R²=0.8193 used proxy labels (linear combination of band powers) derived from the same band-power features used as inputs. Registry experimentId pointed to seed run (m40, R²=0.0) not training run (m43, R²=0.8193). Sleep-EDF PSG annotations not available. BLOCKED until genuine ground-truth data is available.",
  },
};

/**
 * Browser fallback probe: V2-32 (32-D) → sleep quality (1-D).
 *
 * A lightweight linear projection of the 2312-D quality probe, suitable for
 * browser-side inference. Uses 5-band spectral power ratio (delta-theta
 * balance) as a heuristic proxy for sleep depth/quality.
 *
 * Inference: both (browser WASM + server).
 */
export const SLEEP_QUALITY_PROBE_V2_32: TaskHeadDescriptor = {
  id: "sleep-quality-v2-32d",
  name: "Sleep Quality Probe (V2-32 Browser Fallback)",
  version: "0.1.0",
  service: "sleep-staging",
  inputDim: V2_EMBEDDING_DIM,
  outputDim: 1,
  inferenceTarget: "both",
  sha256: "39c624807a9e950b7cf129ede58a84b31741cefb7829ace4c7274b7dc7f3b5fe",
  artifactUri: "/models/sleep/quality-probe-v2-32d-v1.onnx",
  training: {
    dataset: "Sleep-EDF (V2-32 projected)",
    protocol: "5-fold LOSO cross-validation on V2-32 embeddings",
    metrics: {
              r2: -1.6404,
              rmse: 0.1172,
    },
  },
  validation: {
    loso_folds: 5,
    note: "Browser fallback — trained Ridge regression on V2-32 EEGMMIDB embeddings (5-fold LOSO)",
  },
  experimentId: "m38-sleep-quality-blocked",
  scientificStatus: "PROXY_DEMONSTRATION",
  previousMetrics: {
    status: "INVALID",
    reason: "V2-32 fallback R²=-1.6404 worse than mean predictor. Proxy labels. Reclassified per freeze.",
  },
};

/** All sleep task heads (staging + quality), in priority order. */
export const SLEEP_HEADS: TaskHeadDescriptor[] = [
  SLEEP_STAGING_PROBE_JOINT_2312,
  SLEEP_STAGING_PROBE_V2_32,
  SLEEP_QUALITY_PROBE_JOINT_2312,
  SLEEP_QUALITY_PROBE_V2_32,
];

/**
 * Register all sleep task heads into the shared TaskHeadRegistry.
 * Call this at application startup (server) or module load (browser).
 */
export function registerSleepHeads(): void {
  for (const head of SLEEP_HEADS) {
    registerTaskHead(head);
  }
}

/** Convenience: get the default sleep-staging head for a given inference target. */
export function getDefaultSleepHead(
  target: "server" | "browser" | "both" = "server",
): TaskHeadDescriptor | undefined {
  const heads = SLEEP_HEADS.filter(
    (h) => h.inferenceTarget === target || h.inferenceTarget === "both",
  ).filter((h) => h.id.includes("staging"));
  return heads[0];
}

/** Convenience: get the default sleep-quality head for a given inference target. */
export function getDefaultSleepQualityHead(
  target: "server" | "browser" | "both" = "server",
): TaskHeadDescriptor | undefined {
  const heads = SLEEP_HEADS.filter(
    (h) => h.inferenceTarget === target || h.inferenceTarget === "both",
  ).filter((h) => h.id.includes("quality"));
  return heads[0];
}
