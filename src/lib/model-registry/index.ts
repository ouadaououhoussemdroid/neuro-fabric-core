/**
 * Legacy model registry — now derived from the AI layer's authoritative
 * `src/lib/ai/models/registry.ts`.
 *
 * Existing consumers (e.g. the /models UI page) continue to import
 * `getModelsByType`, `getModel`, `MODEL_REGISTRY`, `ACTIVE_EMBEDDER`, and
 * `ACTIVE_DECODER` from this module; the data now flows from a single source
 * of truth so the UI can never drift from the inference contract.
 *
 * - **Embedders** are derived from `listModels()` (the AI registry). Each
 *   descriptor is mapped to the legacy `ModelVersion` shape.
 * - **Decoders** remain defined here because they are not model-artifact
 *   adapters — they are signal-processing heuristics (baseline-spectral-v1)
 *   and ONNX logistic-regression heads (trained-logistic-v0) that live outside
 *   the foundation-model adapter registry.
 */
import { listModels, DEFAULT_EMBEDDER_ID } from "../ai/models/registry";
import type { ModelDescriptor } from "../ai/types";

export type DecoderModelId = "baseline-spectral-v1" | "tfjs-eeg-v1" | "trained-logistic-v0";
// EmbedderModelId is now open-ended: it tracks the AI registry's model IDs
// (pca-legacy-v1, onnx-eegpt, braindecode-eegconformer-prod, …).
export type EmbedderModelId = string;

export interface ModelVersion {
  id: string;
  type: "decoder" | "embedder";
  description: string;
  version: string;
  isExperimental: boolean;
  inputShape: string;
  outputShape: string;
  metrics?: Record<string, number | null>;
  metricsNote?: string;
  createdAt: string;
}

/** Derive a human-readable input shape string from a descriptor's capabilities. */
function descInputShape(d: ModelDescriptor): string {
  const c = d.capabilities;
  if (c.task === "embedding") {
    if (c.channels != null && c.sampleRate != null && c.windowSamples != null) {
      return `[1, ${c.channels}, ${c.windowSamples}] @ ${c.sampleRate}Hz`;
    }
    return "EEGWindow[]";
  }
  return "signal";
}

/** Derive a human-readable output shape string from a descriptor's capabilities. */
function descOutputShape(d: ModelDescriptor): string {
  const c = d.capabilities;
  if (c.embeddingDim != null) return `float[${c.embeddingDim}]`;
  if (c.numClasses != null) return `class[${c.numClasses}]`;
  return "float[]";
}

/** Map an AI-registry descriptor to the legacy ModelVersion shape. */
function toModelVersion(d: ModelDescriptor): ModelVersion {
  return {
    id: d.id,
    type: "embedder",
    description: d.description,
    version: d.version,
    isExperimental: d.isExperimental,
    inputShape: descInputShape(d),
    outputShape: descOutputShape(d),
    createdAt: d.createdAt,
  };
}

/** Decoder entries — not managed by the AI model registry. */
const DECODERS: ModelVersion[] = [
  {
    id: "trained-logistic-v0",
    type: "decoder",
    description:
      "Calibrated logistic regression (5 band-power features → attention/workload/arousal probabilities). " +
      "Exported to ONNX, runs via onnxruntime-web with heuristic fallback. " +
      "Trained on synthetic band-power dataset; val_acc ~0.78 on synthetic validation split.",
    version: "0.1.0",
    isExperimental: false,
    inputShape: "float[5] band-power features",
    outputShape: "{ attention, workload, arousal } ∈ [0,1]",
    metrics: { speed_ms: 3 },
    createdAt: "2026-06-17",
  },
  {
    id: "baseline-spectral-v1",
    type: "decoder",
    description:
      "Spectral ratio heuristics: beta/(alpha+theta) for attention, theta/alpha for workload, " +
      "beta+gamma for arousal. Mathematically grounded but unvalidated against ground truth.",
    version: "1.0.0",
    isExperimental: false,
    inputShape: "EEGSignal",
    outputShape: "{ attention, workload, arousal } ∈ [0,1]",
    metrics: { speed_ms: 2 },
    metricsNote: "No validation metrics — unvalidated heuristic. val_mse not measured.",
    createdAt: "2026-05-25",
  },
  {
    id: "tfjs-eeg-v1",
    type: "decoder",
    description:
      "3-layer MLP (5→32→16→3) with SYNTHETIC weights — not trained on real data. " +
      "Blended 70/30 with spectral baseline. Weights are hand-coded placeholders.",
    version: "0.1.0-synthetic",
    isExperimental: true,
    inputShape: "float[5] band-power features",
    outputShape: "{ attention, workload, arousal } ∈ [0,1]",
    metrics: { speed_ms: 3 },
    metricsNote: "NO validation metrics — weights are synthetic. Do not cite performance numbers.",
    createdAt: "2026-06-07",
  },
];

/**
 * Computed registry: embedders derived from the AI registry + static decoder
 * entries. Recomputed on every access so it always reflects the current
 * AI registry state (e.g. after register/unregisterModel calls).
 */
export function getModelVersions(): ModelVersion[] {
  return [...listModels().map(toModelVersion), ...DECODERS];
}

export function getModelsByType(type: "decoder" | "embedder"): ModelVersion[] {
  return getModelVersions().filter((m) => m.type === type);
}

export function getModel(id: string): ModelVersion | undefined {
  return getModelVersions().find((m) => m.id === id);
}

/**
 * The set of models visible to the UI (kept for backward-compatibility with
 * any code that imports `MODEL_REGISTRY` directly). This is a snapshot —
 * call `getModelVersions()` for the live, derived view.
 */
export const MODEL_REGISTRY: Record<string, ModelVersion> = Object.fromEntries(
  getModelVersions().map((m) => [m.id, m]),
);

/**
 * Active embedder: derived from the AI registry's DEFAULT_EMBEDDER_ID so the
 * UI can never show a stale active model. This is "pca-legacy-v1" — the
 * PCA legacy adapter that embedEEG() falls back to when EEGConformer is gated
 * off (the canonical 32-D contract).
 */
export const ACTIVE_EMBEDDER: EmbedderModelId = DEFAULT_EMBEDDER_ID;

/**
 * Active decoder: "trained-logistic-v0" — the ONNX logistic-regression decoder
 * that decodeCognitiveState() prefers. Falls back to "baseline-spectral-v1"
 * (heuristic) when the ONNX model is unavailable.
 */
export const ACTIVE_DECODER: DecoderModelId = "trained-logistic-v0";
