/**
 * Legacy PCA / linear-AE adapter. Wraps the existing embedSignal pipeline so
 * the foundation-model layer has a working default without touching legacy
 * callers.
 */
import { embedSignal } from "../../embeddings";
import { segment } from "../../eeg/preprocessing/segment";
import type { EEGModelAdapter } from "./types";
import type { EmbeddingOutput, ModelDescriptor, ModelInput } from "../types";

/** Pad with zeros or truncate to exactly `dim` elements. */
function padOrTruncate(vec: number[], dim: number): number[] {
  if (vec.length === dim) return vec;
  if (vec.length > dim) return vec.slice(0, dim);
  const out = new Array<number>(dim).fill(0);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i];
  return out;
}

export class PCAEmbeddingAdapter implements EEGModelAdapter {
  readonly descriptor: ModelDescriptor = {
    id: "pca-legacy-v1",
    kind: "linear-ae",
    name: "PCA / Linear Autoencoder (Legacy)",
    version: "1.0.0",
    description:
      "Band-power features projected through a closed-form linear autoencoder (PCA). Default fallback embedder.",
    isExperimental: false,
    capabilities: {
      task: "embedding",
      channels: null,
      sampleRate: null,
      windowSamples: null,
      // Canonical 32-D contract: matches vector(32) + EEGConformer head.
      embeddingDim: 32,
      runtime: "js",
      implemented: true,
    },
    createdAt: "2026-05-25",
  };

  private loaded = false;

  async load(): Promise<void> {
    this.loaded = true;
  }

  async unload(): Promise<void> {
    this.loaded = false;
  }

  isLoaded(): boolean {
    return this.loaded;
  }

  async embed(input: ModelInput): Promise<EmbeddingOutput> {
    const windows =
      input.kind === "windows"
        ? input.windows
        : input.kind === "signal"
          ? segment(input.signal.data, input.signal.sampleRate, 2, 0.5)
          : (() => {
              throw new Error("PCAEmbeddingAdapter: feature input not supported");
            })();
    const res = embedSignal(windows, this.descriptor.capabilities.embeddingDim ?? 32);
    // PCA can return fewer dims than embeddingDim when featureDim < latentDim
    // (e.g. low-channel inputs). Pad/truncate to exactly embeddingDim so
    // producer dim == DB vector(32) dim on every upload.
    const targetDim = this.descriptor.capabilities.embeddingDim ?? 32;
    const vector = padOrTruncate(res.vector, targetDim);
    return {
      vector,
      dim: vector.length,
      modelId: this.descriptor.id,
      durationMs: res.durationMs,
    };
  }
}
