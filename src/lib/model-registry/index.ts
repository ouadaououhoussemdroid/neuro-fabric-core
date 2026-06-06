export type DecoderModelId = "baseline-spectral-v1" | "tfjs-eeg-v1";
export type EmbedderModelId = "raw-bandpower" | "linear-ae" | "tfjs-autoencoder-v1";

export interface ModelVersion {
  id: string; type: "decoder" | "embedder"; description: string; version: string;
  isExperimental: boolean; inputShape: string; outputShape: string;
  metrics?: Record<string, number>; createdAt: string;
}

export const MODEL_REGISTRY: Record<string, ModelVersion> = {
  "baseline-spectral-v1": { id: "baseline-spectral-v1", type: "decoder", description: "Spectral ratio heuristics (beta/alpha+theta for attention, theta/alpha for workload)", version: "1.0.0", isExperimental: false, inputShape: "EEGSignal", outputShape: "{ attention, workload, arousal } ∈ [0,1]", metrics: { interpretability: 1.0, speed_ms: 2 }, createdAt: "2026-05-25" },
  "raw-bandpower": { id: "raw-bandpower", type: "embedder", description: "Raw band-power features without dimensionality reduction (fallback)", version: "1.0.0", isExperimental: false, inputShape: "EEGWindow[]", outputShape: "float[]", metrics: { speed_ms: 1 }, createdAt: "2026-05-25" },
  "linear-ae": { id: "linear-ae", type: "embedder", description: "Linear autoencoder (PCA projection) to latent dim", version: "1.0.0", isExperimental: false, inputShape: "EEGWindow[]", outputShape: "float[latentDim]", metrics: { speed_ms: 5 }, createdAt: "2026-05-25" },
  "tfjs-eeg-v1": { id: "tfjs-eeg-v1", type: "decoder", description: "TensorFlow.js neural decoder — trained on PhysioNet EEG data (coming soon)", version: "0.1.0", isExperimental: true, inputShape: "float[C × W]", outputShape: "{ attention, workload, arousal } ∈ [0,1]", metrics: {}, createdAt: "2026-06-07" },
  "tfjs-autoencoder-v1": { id: "tfjs-autoencoder-v1", type: "embedder", description: "Deep convolutional autoencoder — TensorFlow.js (coming soon)", version: "0.1.0", isExperimental: true, inputShape: "float[C × W]", outputShape: "float[128]", metrics: {}, createdAt: "2026-06-07" },
};

export function getModelsByType(type: "decoder" | "embedder"): ModelVersion[] {
  return Object.values(MODEL_REGISTRY).filter((m) => m.type === type);
}

export function getModel(id: string): ModelVersion | undefined {
  return MODEL_REGISTRY[id];
}

export const ACTIVE_DECODER: DecoderModelId = "baseline-spectral-v1";
export const ACTIVE_EMBEDDER: EmbedderModelId = "linear-ae";
