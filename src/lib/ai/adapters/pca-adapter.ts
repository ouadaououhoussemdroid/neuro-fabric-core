/**
 * Legacy PCA / linear-AE adapter. Wraps the existing embedSignal pipeline so
 * the foundation-model layer has a working default without touching legacy
 * callers.
 */
import { embedSignal } from "../../embeddings";
import { segmentWindows } from "../../eeg/preprocessing/segment";
import type { EEGModelAdapter } from "./types";
import type { EmbeddingOutput, ModelDescriptor, ModelInput } from "../types";

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
      embeddingDim: 64,
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
          ? segmentWindows(input.signal, { windowSec: 2, hopSec: 1 })
          : (() => {
              throw new Error("PCAEmbeddingAdapter: feature input not supported");
            })();
    const res = embedSignal(windows, this.descriptor.capabilities.embeddingDim ?? 64);
    return {
      vector: res.vector,
      dim: res.dimensions,
      modelId: this.descriptor.id,
      durationMs: res.durationMs,
    };
  }
}
