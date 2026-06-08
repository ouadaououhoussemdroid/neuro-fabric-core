export type DecoderModelId = "baseline-spectral-v1" | "tfjs-eeg-v1";
export type EmbedderModelId = "raw-bandpower" | "linear-ae" | "tfjs-autoencoder-v1";

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

export const MODEL_REGISTRY: Record<string, ModelVersion> = {
  "baseline-spectral-v1": {
    id: "baseline-spectral-v1",
    type: "decoder",
    description: "Spectral ratio heuristics: beta/(alpha+theta) for attention, theta/alpha for workload, beta+gamma for arousal. Mathematically grounded but unvalidated against ground truth.",
    version: "1.0.0",
    isExperimental: false,
    inputShape: "EEGSignal",
    outputShape: "{ attention, workload, arousal } ∈ [0,1]",
    metrics: { speed_ms: 2 },
    metricsNote: "No validation metrics — unvalidated heuristic. val_mse not measured.",
    createdAt: "2026-05-25",
  },
  "tfjs-eeg-v1": {
    id: "tfjs-eeg-v1",
    type: "decoder",
    description: "3-layer MLP (5→32→16→3) with SYNTHETIC weights — not trained on real data. Blended 70/30 with spectral baseline. Weights are hand-coded placeholders pending real training.",
    version: "0.1.0-synthetic",
    isExperimental: true,
    inputShape: "float[5] band-power features",
    outputShape: "{ attention, workload, arousal } ∈ [0,1]",
    metrics: { speed_ms: 3 },
    metricsNote: "NO validation metrics — weights are synthetic. Do not cite performance numbers.",
    createdAt: "2026-06-07",
  },
  "raw-bandpower": {
    id: "raw-bandpower",
    type: "embedder",
    description: "Raw band-power features without dimensionality reduction (fallback)",
    version: "1.0.0",
    isExperimental: false,
    inputShape: "EEGWindow[]",
    outputShape: "float[]",
    metrics: { speed_ms: 1 },
    createdAt: "2026-05-25",
  },
  "linear-ae": {
    id: "linear-ae",
    type: "embedder",
    description: "Linear autoencoder — mathematically equivalent to PCA. Closed-form solution, no training required.",
    version: "1.0.0",
    isExperimental: false,
    inputShape: "EEGWindow[]",
    outputShape: "float[latentDim]",
    metrics: { speed_ms: 5 },
    createdAt: "2026-05-25",
  },
  "tfjs-autoencoder-v1": {
    id: "tfjs-autoencoder-v1",
    type: "embedder",
    description: "Deep convolutional autoencoder — planned, not yet implemented.",
    version: "0.1.0",
    isExperimental: true,
    inputShape: "float[C × W]",
    outputShape: "float[128]",
    metrics: {},
    metricsNote: "Not implemented.",
    createdAt: "2026-06-07",
  },
};

export function getModelsByType(type: "decoder" | "embedder"): ModelVersion[] {
  return Object.values(MODEL_REGISTRY).filter((m) => m.type === type);
}

export function getModel(id: string): ModelVersion | undefined {
  return MODEL_REGISTRY[id];
}

// Active: baseline-spectral-v1 is the only scientifically honest decoder
// tfjs-eeg-v1 is experimental with synthetic weights — not used by default
export const ACTIVE_DECODER: DecoderModelId = "baseline-spectral-v1";
export const ACTIVE_EMBEDDER: EmbedderModelId = "linear-ae";
