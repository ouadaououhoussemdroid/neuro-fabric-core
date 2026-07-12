/**
 * T-018 — Saliency map types and loader for EEGConformer (Captum).
 *
 * The actual integrated-gradients computation happens in
 * `scripts/compute_saliency.py` (Captum requires PyTorch). This module
 * provides the types and loader for the JSON sidecar emitted by that
 * script, so the `/embeddings` route can overlay per-channel saliency
 * on a topomap.
 */

/** JSON sidecar emitted by `scripts/compute_saliency.py`. */
export interface SaliencySidecar {
  /** SHA-256 hash of the ONNX artefact (matches T-009 manifest). */
  artefactHash: string;
  /** Path to the .npz file with full attribution maps. */
  saliencyPath: string;
  /** Number of attributed windows. */
  nSamples: number;
  /** Number of channels. */
  channels: number;
  /** Number of samples per window. */
  samples: number;
  /** Per-channel mean saliency (for topomap overlay). Length = channels. */
  channelSaliency: number[];
}

/** Load and validate a saliency sidecar from a parsed JSON object. */
export function parseSaliencySidecar(raw: unknown): SaliencySidecar | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  if (
    typeof obj.artefact_hash !== "string" ||
    typeof obj.saliency_path !== "string" ||
    typeof obj.n_samples !== "number" ||
    typeof obj.channels !== "number" ||
    typeof obj.samples !== "number" ||
    !Array.isArray(obj.channel_saliency)
  ) {
    return null;
  }
  return {
    artefactHash: obj.artefact_hash,
    saliencyPath: obj.saliency_path,
    nSamples: obj.n_samples,
    channels: obj.channels,
    samples: obj.samples,
    channelSaliency: (obj.channel_saliency as unknown[]).map(Number),
  };
}

/**
 * Normalize channel saliency to [0, 1] for visualization (topomap overlay).
 */
export function normalizeSaliency(channelSaliency: number[]): number[] {
  if (channelSaliency.length === 0) return [];
  const max = Math.max(...channelSaliency);
  const min = Math.min(...channelSaliency);
  const range = max - min;
  if (range < 1e-12) return channelSaliency.map(() => 0.5);
  return channelSaliency.map((v) => (v - min) / range);
}
