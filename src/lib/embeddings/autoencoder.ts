/**
 * Linear autoencoder scaffold trained by closed-form least squares against
 * the PCA reconstruction objective. Encoder is a learned projection W (k x d),
 * decoder is W^T. This serves as a starting point for replacing with a
 * non-linear AE backed by an ML runtime once one is available in the worker.
 *
 * For an MVP the linear AE reduces exactly to PCA, but the API surface is
 * shaped so it can be swapped for a deep encoder later without churning callers.
 */
import { fitPCA, transformPCA, type PCAModel } from "./pca";

export interface AutoencoderModel {
  kind: "linear-ae";
  latentDim: number;
  encoder: number[][]; // [k][d]
  decoder: number[][]; // [d][k]
  mean: number[];
}

export function fitAutoencoder(X: number[][], latentDim: number): AutoencoderModel {
  const pca: PCAModel = fitPCA(X, latentDim);
  const encoder = pca.components;
  const decoder: number[][] = Array.from({ length: pca.mean.length }, (_, i) =>
    encoder.map((c) => c[i]),
  );
  return { kind: "linear-ae", latentDim, encoder, decoder, mean: pca.mean };
}

export function encode(model: AutoencoderModel, x: number[]): number[] {
  return transformPCA({ mean: model.mean, components: model.encoder, explainedVariance: [] }, x);
}

export function decode(model: AutoencoderModel, z: number[]): number[] {
  return model.decoder.map((row, i) => {
    let s = 0;
    for (let j = 0; j < row.length; j++) s += row[j] * z[j];
    return s + model.mean[i];
  });
}