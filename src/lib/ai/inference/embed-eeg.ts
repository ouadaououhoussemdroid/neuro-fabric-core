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
 *
 * P3: the preferred (non-PCA) model is now routed through the process-wide
 * `InferenceEngine`, which caches a single ONNX InferenceSession per model and
 * reuses it across requests. This amortises the per-call fetch+compile+worker-
 * init cost that dominated Firefox latency (P95 1589 ms → 162 ms) and clears
 * the GA latency gate on the canonical FP32 artifact (sha 18644de1…).
 */
import { embed, type EmbedResult, finalize } from "../embeddings";
import { hasModel } from "../models/registry";
import { inferenceEngine } from "./engine";
import type { ModelInput } from "../types";
import { log } from "../../logging";
import { isEEGConformerEnabledForUser } from "../rollout";
import { metrics } from "../../metrics";

export interface EmbedEEGOptions {
  /** Preferred EEG foundation model id. Defaults to Braindecode ONNX export. */
  preferredModelId?: string;
  /** Generic ONNX embedder id to try before falling back to PCA. */
  onnxModelId?: string;
  /** L2-normalise. Defaults true so output plugs into cosine search. */
  normalize?: boolean;
  /** Validate against this dim if known up-front. */
  expectedDim?: number;
  /** Authenticated user id for per-user cohort routing. */
  userId?: string;
}

const DEFAULT_PREFERRED = "braindecode-eegconformer-prod-v2";

export async function embedEEG(
  input: ModelInput,
  opts: EmbedEEGOptions = {},
): Promise<EmbedResult> {
  const preferred = opts.preferredModelId ?? DEFAULT_PREFERRED;
  const chain: string[] = [];
  if (opts.onnxModelId && hasModel(opts.onnxModelId)) chain.push(opts.onnxModelId);
  chain.push("pca-legacy-v1");

  const isEEGConformer = preferred === DEFAULT_PREFERRED;
  const enabled = isEEGConformer ? isEEGConformerEnabledForUser(opts.userId) : true;

  // T-016 — Canary observability: record cohort eligibility.
  if (isEEGConformer) {
    if (enabled) {
      metrics.cohortChecksTotal.inc({ result: "hit" });
    } else {
      metrics.cohortChecksTotal.inc({ result: "miss" });
    }
  }

  const normalize = opts.normalize !== false;

  // P3: route the preferred non-PCA model through the cached InferenceEngine
  // so a single InferenceSession is created and reused (amortises the
  // fetch + verify + compile + worker-init cost). The engine's per-model
  // async mutex serializes session.run() (ORT-Web WASM is not reentrant).
  const startId = enabled && hasModel(preferred) ? preferred : chain[0];
  log("info", "ai.embedEEG.start", { startId, chain });

  if (startId !== "pca-legacy-v1") {
    try {
      const out = await inferenceEngine.embed(startId, input);
      const result = finalize(out, false, undefined, normalize, opts.expectedDim);
      metrics.modelSelectedTotal.inc({ model: result.modelId, fell_back: "false" });
      return result;
    } catch (err) {
      const reason = (err as Error).message;
      log("warn", "ai.embedEEG.engine.fail", { modelId: startId, reason });
      // Evict the broken session so the next request retries a fresh one.
      await inferenceEngine.disposeModel(startId);
      // Fall through to per-call facade → re-verify SHA → fallbackChain → PCA.
    }
  }

  const result = await embed(input, {
    modelId: startId,
    fallbackChain: chain,
    fallbackToPCA: true,
    normalize,
    expectedDim: opts.expectedDim,
  });

  // T-016 — Canary observability: record which model actually produced the
  // embedding, including whether it fell back from the requested model.
  const selectedModel = result.modelId;
  metrics.modelSelectedTotal.inc({ model: selectedModel, fell_back: String(result.fellBack) });

  return result;
}

export { DEFAULT_PREFERRED }; // keep export at end for readability
