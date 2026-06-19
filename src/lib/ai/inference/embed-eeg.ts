/**
 * High-level EEG embedding entry point. Implements the production-grade
 * fallback chain:
 *
 *   Chosen EEG foundation model (Braindecode-ONNX)
 *     ↓ on failure / unavailable
 *   Generic ONNX embedder (if registered)
 *     ↓
 *   PCA legacy embedder (always available)
 *
 * Validation (NaN, dim, zero-vector) and L2 normalisation are handled by the
 * underlying `embed()` facade; we only assemble the chain + observability.
 */
import { embed, type EmbedResult } from "../embeddings";
import { hasModel } from "../models/registry";
import type { ModelInput } from "../types";
import { log } from "../../logging";

export interface EmbedEEGOptions {
  /** Preferred EEG foundation model id. Defaults to Braindecode ONNX export. */
  preferredModelId?: string;
  /** Generic ONNX embedder id to try before falling back to PCA. */
  onnxModelId?: string;
  /** L2-normalise. Defaults true so output plugs into cosine search. */
  normalize?: boolean;
  /** Validate against this dim if known up-front. */
  expectedDim?: number;
}

const DEFAULT_PREFERRED = "braindecode-eegconformer-prod";

export async function embedEEG(
  input: ModelInput,
  opts: EmbedEEGOptions = {},
): Promise<EmbedResult> {
  const preferred = opts.preferredModelId ?? DEFAULT_PREFERRED;
  const chain: string[] = [];
  if (opts.onnxModelId && hasModel(opts.onnxModelId)) chain.push(opts.onnxModelId);
  chain.push("pca-legacy-v1");

  const startId = hasModel(preferred) ? preferred : chain[0];
  log("info", "ai.embedEEG.start", { startId, chain });

  return embed(input, {
    modelId: startId,
    fallbackChain: chain,
    fallbackToPCA: true,
    normalize: opts.normalize !== false,
    expectedDim: opts.expectedDim,
  });
}