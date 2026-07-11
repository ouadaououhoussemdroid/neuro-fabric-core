/**
 * Foundation-model embeddings facade. Routes through the registered adapter
 * for `modelId`, defaulting to the PCA legacy adapter so existing callers see
 * no behavior change. Falls back to PCA automatically on adapter failure.
 */
import { createAdapter, DEFAULT_EMBEDDER_ID, hasModel } from "../models/registry";
import type { EmbeddingOutput, ModelInput } from "../types";
import { log } from "../../logging";
import { l2Normalize, validateEmbedding, EmbeddingValidationError } from "../validation";

/**
 * Loud, observable signal that the embedding pipeline degraded to the PCA
 * baseline instead of the requested neural model. UIs can listen on
 * `window` (CustomEvent of this name, `detail: { modelId, reason, requested }`)
 * to render a visible badge so a fallback is never silent in production.
 */
export const EMBED_FALLBACK_EVENT = "neurofabric:embed-fallback";

function announceFallback(detail: {
  requestedModelId: string;
  resolvedModelId: string;
  reason: string;
}) {
  // Loud console signal — visible in production browser devtools and server logs.

  console.error(
    `[neurofabric] EEG embedding fell back to PCA baseline (${detail.resolvedModelId}). ` +
      `Requested "${detail.requestedModelId}". Reason: ${detail.reason}`,
  );
  log("error", "ai.embed.fallback.loud", detail);
  if (typeof window !== "undefined" && typeof CustomEvent === "function") {
    try {
      window.dispatchEvent(new CustomEvent(EMBED_FALLBACK_EVENT, { detail }));
    } catch {
      /* ignore — non-DOM env */
    }
  }
}

export interface EmbedOptions {
  modelId?: string;
  /** When true and the requested adapter fails, fall back to PCA. Default: true. */
  fallbackToPCA?: boolean;
  /** L2-normalise the output. Default: true (so vectors plug into cosine search). */
  normalize?: boolean;
  /** Override the validation contract; falls back to artifact metadata. */
  expectedDim?: number;
  /**
   * Explicit fallback chain tried in order if `modelId` fails. When omitted,
   * the chain defaults to `[ "pca-legacy-v1" ]` (preserves previous
   * behaviour). Pass e.g. `[ "onnx-...", "pca-legacy-v1" ]` to cascade
   * Braindecode → ONNX → PCA.
   */
  fallbackChain?: string[];
}

export interface EmbedResult extends EmbeddingOutput {
  fellBack: boolean;
  fallbackReason?: string;
  normalized: boolean;
}

export async function embed(input: ModelInput, opts: EmbedOptions = {}): Promise<EmbedResult> {
  const id = opts.modelId ?? DEFAULT_EMBEDDER_ID;
  const fallback = opts.fallbackToPCA !== false;
  const normalize = opts.normalize !== false;
  const chain = opts.fallbackChain ?? [DEFAULT_EMBEDDER_ID];
  log("info", "ai.embed.start", { modelId: id, normalize, fallback });
  if (!hasModel(id)) {
    if (!fallback) throw new Error(`Unknown model id: ${id}`);
    return runFallbackChain(input, `unknown model "${id}"`, normalize, opts.expectedDim, chain);
  }
  const adapter = createAdapter(id);
  if (!adapter.embed) {
    if (!fallback) throw new Error(`Adapter "${id}" does not support embeddings`);
    return runFallbackChain(
      input,
      `adapter "${id}" has no embed()`,
      normalize,
      opts.expectedDim,
      chain,
    );
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
    return runFallbackChain(input, reason, normalize, opts.expectedDim, chain, id);
  } finally {
    try {
      await adapter.unload();
    } catch {
      /* noop */
    }
  }
}

async function runFallbackChain(
  input: ModelInput,
  initialReason: string,
  normalize: boolean,
  expectedDim: number | undefined,
  chain: string[],
  requestedModelId: string = chain[0] ?? DEFAULT_EMBEDDER_ID,
): Promise<EmbedResult> {
  const reasons: string[] = [initialReason];
  const tried = new Set<string>();
  // Always guarantee PCA at the tail so the chain terminates.
  const ordered = [...chain.filter((id) => id !== DEFAULT_EMBEDDER_ID), DEFAULT_EMBEDDER_ID];
  for (const id of ordered) {
    if (tried.has(id)) continue;
    tried.add(id);
    log("warn", "ai.embed.fallback.try", { modelId: id, prevReason: reasons.at(-1) });
    if (!hasModel(id)) {
      reasons.push(`unknown model "${id}"`);
      continue;
    }
    const adapter = createAdapter(id);
    if (!adapter.embed) {
      reasons.push(`adapter "${id}" has no embed()`);
      continue;
    }
    try {
      await adapter.load();
      const out = await adapter.embed(input);
      const joined = reasons.join(" → ");
      announceFallback({
        requestedModelId,
        resolvedModelId: out.modelId,
        reason: joined,
      });
      return finalize(out, true, joined, normalize, expectedDim);
    } catch (err) {
      reasons.push(`${id}: ${(err as Error).message}`);
    } finally {
      try {
        await adapter.unload();
      } catch {
        /* noop */
      }
    }
  }
  throw new Error(`embed: all adapters failed (${reasons.join(" → ")})`);
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
