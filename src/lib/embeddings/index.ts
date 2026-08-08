import type { EEGWindow } from "../eeg/types";
import { bandPowerFeatures } from "./features";
import { fitPCA, transformPCA, type PCAModel } from "./pca";
import { fitAutoencoder, encode as aeEncode, type AutoencoderModel } from "./autoencoder";

export { bandPowerFeatures, fitPCA, transformPCA, fitAutoencoder, aeEncode };
export type { PCAModel, AutoencoderModel };

export interface EmbeddingResult {
  vector: number[];
  dimensions: number;
  featureDim: number;
  durationMs: number;
  model: "pca" | "linear-ae" | "raw-bandpower";
  fellBack: boolean;
}

/** Extract features from each window and stack into a feature matrix. */
export function extractFeatureMatrix(windows: EEGWindow[]): number[][] {
  return windows.map(bandPowerFeatures);
}

/**
 * Embed a sequence of EEG windows into a fixed-length vector.
 *
 * Strategy (fallback chain):
 *   1. Linear autoencoder → "linear-ae" (when enough samples for a stable fit)
 *   2. PCA projection      → "pca"       (when enough samples for PCA but too
 *      few for a stable AE fit)
 *   3. Raw band-power      → "raw-bandpower" (last resort, no learned projection)
 *
 * `fellBack` is `true` whenever the pipeline could not use the primary method
 * and degraded to a simpler embedding.
 *
 * Default `latentDim` is 32 — the canonical embedding dimension that matches the
 * `embeddings.embedding vector(32)` pgvector contract and the EEGConformer
 * output head. PCA callers pass this default; the PCA adapter pads/truncates
 * the result to exactly this width so producer dim == database dim.
 */
export function embedSignal(windows: EEGWindow[], latentDim = 32): EmbeddingResult {
  if (windows.length === 0) {
    throw new Error("embedSignal: no windows");
  }
  const t0 = performance.now();
  const features = extractFeatureMatrix(windows);
  const featureDim = features[0].length;

  // Mean-pool across windows into a single feature vector.
  const pooled = new Array<number>(featureDim).fill(0);
  for (const f of features) for (let i = 0; i < featureDim; i++) pooled[i] += f[i];
  for (let i = 0; i < featureDim; i++) pooled[i] /= features.length;

  // Primary path: linear autoencoder (needs at least latentDim+1 samples).
  if (features.length >= Math.max(latentDim + 1, 4) && featureDim > latentDim) {
    const ae: AutoencoderModel = fitAutoencoder(features, latentDim);
    const z = aeEncode(ae, pooled);
    return {
      vector: z,
      dimensions: z.length,
      featureDim,
      durationMs: +(performance.now() - t0).toFixed(2),
      model: "linear-ae",
      fellBack: false,
    };
  }

  // Fallback 1: PCA (needs at least min(latentDim, featureDim) samples).
  if (features.length >= 4) {
    const k = Math.min(latentDim, featureDim);
    const pca = fitPCA(features, k);
    const z = transformPCA(pca, pooled);
    return {
      vector: z,
      dimensions: z.length,
      featureDim,
      durationMs: +(performance.now() - t0).toFixed(2),
      model: "pca",
      fellBack: true,
    };
  }

  // Fallback 2: raw band-power (no learned projection).
  return {
    vector: pooled,
    dimensions: featureDim,
    featureDim,
    durationMs: +(performance.now() - t0).toFixed(2),
    model: "raw-bandpower",
    fellBack: true,
  };
}
