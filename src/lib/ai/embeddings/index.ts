/**
 * Foundation-model embeddings facade. Routes through the registered adapter
 * for `modelId`, defaulting to the PCA legacy adapter so existing callers see
 * no behavior change. Falls back to PCA automatically on adapter failure.
 */
import { createAdapter, DEFAULT_EMBEDDER_ID, hasModel } from "../models/registry";
import type { EmbeddingOutput, ModelInput } from "../types";

export interface EmbedOptions {
  modelId?: string;
  /** When true and the requested adapter fails, fall back to PCA. Default: true. */
  fallbackToPCA?: boolean;
}

export interface EmbedResult extends EmbeddingOutput {
  fellBack: boolean;
  fallbackReason?: string;
}

export async function embed(
  input: ModelInput,
  opts: EmbedOptions = {},
): Promise<EmbedResult> {
  const id = opts.modelId ?? DEFAULT_EMBEDDER_ID;
  const fallback = opts.fallbackToPCA !== false;
  if (!hasModel(id)) {
    if (!fallback) throw new Error(`Unknown model id: ${id}`);
    return runFallback(input, `unknown model "${id}"`);
  }
  const adapter = createAdapter(id);
  if (!adapter.embed) {
    if (!fallback) throw new Error(`Adapter "${id}" does not support embeddings`);
    return runFallback(input, `adapter "${id}" has no embed()`);
  }
  try {
    await adapter.load();
    const out = await adapter.embed(input);
    return { ...out, fellBack: false };
  } catch (err) {
    if (!fallback) throw err;
    return runFallback(input, (err as Error).message);
  } finally {
    try { await adapter.unload(); } catch { /* noop */ }
  }
}

async function runFallback(input: ModelInput, reason: string): Promise<EmbedResult> {
  const adapter = createAdapter(DEFAULT_EMBEDDER_ID);
  await adapter.load();
  try {
    const out = await adapter.embed!(input);
    return { ...out, fellBack: true, fallbackReason: reason };
  } finally {
    await adapter.unload();
  }
}

export { DEFAULT_EMBEDDER_ID };