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
}

/** Extract features from each window and stack into a feature matrix. */
export function extractFeatureMatrix(windows: EEGWindow[]): number[][] {
  return windows.map(bandPowerFeatures);
}

/**
 * Pool a per-window feature matrix into a single embedding by averaging across
 * windows, then optionally projecting through PCA. When the matrix has fewer
 * rows than `latentDim`, PCA is skipped and the raw averaged feature vector is
 * returned (still real, just no learned projection).
 */
export function embedSignal(windows: EEGWindow[], latentDim = 64): EmbeddingResult {
  if (windows.length === 0) {
    throw new Error("embedSignal: no windows");
  }
  const t0 = performance.now();
  const features = extractFeatureMatrix(windows);
  const featureDim = features[0].length;

  // Mean-pool across windows
  const pooled = new Array<number>(featureDim).fill(0);
  for (const f of features) for (let i = 0; i < featureDim; i++) pooled[i] += f[i];
  for (let i = 0; i < featureDim; i++) pooled[i] /= features.length;

  if (features.length < Math.max(latentDim, 4) || featureDim <= latentDim) {
    return {
      vector: pooled,
      dimensions: featureDim,
      featureDim,
      durationMs: +(performance.now() - t0).toFixed(2),
      model: "raw-bandpower",
    };
  }

  const ae: AutoencoderModel = fitAutoencoder(features, latentDim);
  const z = aeEncode(ae, pooled);
  return {
    vector: z,
    dimensions: z.length,
    featureDim,
    durationMs: +(performance.now() - t0).toFixed(2),
    model: "linear-ae",
  };
}
