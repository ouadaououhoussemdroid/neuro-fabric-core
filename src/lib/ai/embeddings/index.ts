/**
 * Foundation-model embeddings facade. Routes through the registered adapter
 * for `modelId`, defaulting to the PCA legacy adapter so existing callers see
 * no behavior change. Falls back to PCA automatically on adapter failure.
 */
import { createAdapter, DEFAULT_EMBEDDER_ID, hasModel } from "../models/registry";
import type { EmbeddingOutput, ModelInput } from "../types";
import { log } from "../../logging";
import {
  l2Normalize,
  validateEmbedding,
  EmbeddingValidationError,
} from "../validation";

export interface EmbedOptions {
  modelId?: string;
  /** When true and the requested adapter fails, fall back to PCA. Default: true. */
  fallbackToPCA?: boolean;
  /** L2-normalise the output. Default: true (so vectors plug into cosine search). */
  normalize?: boolean;
  /** Override the validation contract; falls back to artifact metadata. */
  expectedDim?: number;
}

export interface EmbedResult extends EmbeddingOutput {
  fellBack: boolean;
  fallbackReason?: string;
  normalized: boolean;
}

export async function embed(
  input: ModelInput,
  opts: EmbedOptions = {},
): Promise<EmbedResult> {
  const id = opts.modelId ?? DEFAULT_EMBEDDER_ID;
  const fallback = opts.fallbackToPCA !== false;
  const normalize = opts.normalize !== false;
  log("info", "ai.embed.start", { modelId: id, normalize, fallback });
  if (!hasModel(id)) {
    if (!fallback) throw new Error(`Unknown model id: ${id}`);
    return runFallback(input, `unknown model "${id}"`, normalize, opts.expectedDim);
  }
  const adapter = createAdapter(id);
  if (!adapter.embed) {
    if (!fallback) throw new Error(`Adapter "${id}" does not support embeddings`);
    return runFallback(input, `adapter "${id}" has no embed()`, normalize, opts.expectedDim);
  }
  try {
    log("info", "ai.embed.load", { modelId: id });
    await adapter.load();
    const out = await adapter.embed(input);
    return finalize(out, false, undefined, normalize, opts.expectedDim);
  } catch (err) {
    const reason = (err as Error).message;
    log("warn", "ai.embed.fail", { modelId: id, reason });
    if (!fallback) throw err;
    return runFallback(input, reason, normalize, opts.expectedDim);
  } finally {
    try { await adapter.unload(); } catch { /* noop */ }
  }
}

async function runFallback(
  input: ModelInput,
  reason: string,
  normalize: boolean,
  expectedDim: number | undefined,
): Promise<EmbedResult> {
  log("warn", "ai.embed.fallback", { reason });
  const adapter = createAdapter(DEFAULT_EMBEDDER_ID);
  await adapter.load();
  try {
    const out = await adapter.embed!(input);
    return finalize(out, true, reason, normalize, expectedDim);
  } finally {
    await adapter.unload();
  }
}

function finalize(
  out: EmbeddingOutput,
  fellBack: boolean,
  reason: string | undefined,
  normalize: boolean,
  expectedDim: number | undefined,
): EmbedResult {
  try {
    validateEmbedding(out.vector, { expectedDim });
  } catch (e) {
    if (e instanceof EmbeddingValidationError) {
      log("error", "ai.embed.invalid", { modelId: out.modelId, code: e.code });
    }
    throw e;
  }
  const vector = normalize ? l2Normalize(out.vector) : out.vector;
  log("info", "ai.embed.done", {
    modelId: out.modelId,
    dim: vector.length,
    durationMs: out.durationMs,
    fellBack,
    normalized: normalize,
  });
  return {
    ...out,
    vector,
    dim: vector.length,
    fellBack,
    fallbackReason: reason,
    normalized: normalize,
  };
}

export { DEFAULT_EMBEDDER_ID };