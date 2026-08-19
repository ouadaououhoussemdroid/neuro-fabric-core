/**
 * T-036 / Mission 12 — Tier-2 server-native CBraMod 200-D foundation embedder.
 *
 * WHY THIS IS SERVER-ONLY: CBraMod's ONNX uses `DFT` and `ReduceL2` ops, which
 * ORT-WASM does not implement (manifest: wasmCompatible:false). It therefore
 * CANNOT run in the browser / on ORT-Web / in Node's WASM backend — it runs only
 * on the server via the native `onnxruntime-node` CPU EP. This module is
 * suffixed `.server.ts` (Vinxi/TanStack Start convention) so the browser bundle
 * never includes it, and `onnxruntime-node` is imported DYNAMICALLY inside
 * `foundationRuntime()`, so non-node clients never load the native addon.
 *
 * PIPELINE (mirrors Mission-11 preprocessing/pooling/provenance exactly):
 *   EEG[64ch/160Hz] → selectCbraModChannels(19) → resampleSignal(250)
 *   → preprocess({ bandpass:[4,38], zscore, segment:{4s, 0.5} }) → 4s windows
 *   → ONNXAdapter(onnxruntime-node CPU EP, cbramod-encoder.onnx [1,19,1000]→[1,19,5,200])
 *   → applyOutputPooling("mean-tokens", 200) ≡ Mission-11 `r.mean(axis=(1,2))`
 *   → finalize(validateEmbedding{200} + L2 normalize) → 200-D
 *   → foundation_embeddings table via match_foundation_embeddings RPC.
 *
 * NO FALLBACK BY DESIGN: Tier-2 CBraMod is opt-in only. If the runtime, the
 * artifact, or a single window fails, this throws (never degrades to V2 or PCA).
 * The HTTP layer maps runtime/artifact unavailability to 424 and per-window
 * errors to 500. V2 / DEFAULT_PREFERRED / embedEEG / vector(32) are never
 * imported or called from here.
 */
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

import { ONNXAdapter } from "../adapters/onnx-adapter";
import type { OrtRuntime, OrtSessionLike } from "../adapters/onnx-adapter";
import { finalize } from "../embeddings";
import type { EmbedResult } from "../embeddings";
import {
  verifyArtefact,
  type ArtefactManifest,
  type ArtefactManifestEntry,
} from "../artefacts/hashed-artefact";
import type { ModelInput } from "../types";
import type { EEGWindow } from "../../eeg/types";
import { log } from "../../logging";
import { metrics } from "../../metrics";

/** Tier-2 model id surfaced in embeddings provenance / DB model_id tag. */
export const FOUNDATION_MODEL_ID = "onnx-cbramod-foundation-200d";
/** Manifest artifact id for the CBraMod ONNX weights. */
export const FOUNDATION_ARTIFACT_ID = "cbramod-encoder";
/** CBraMod native output dimension after mean-tokens pooling. */
export const FOUNDATION_EMBEDDING_DIM = 200;
/** CBraMod was trained on 250 Hz, 4 s (1000-sample) windows. */
export const FOUNDATION_SAMPLE_RATE_HZ = 250;
export const FOUNDATION_WINDOW_SAMPLES = 1000;
export const FOUNDATION_CHANNELS = 19;

/** Raised when the native CBraMod runtime or artifact is unavailable. */
export class FoundationUnavailableError extends Error {
  constructor(public readonly reason: string) {
    super(`CBraMod foundation runtime unavailable: ${reason}`);
    this.name = "FoundationUnavailableError";
  }
}

/**
 * Provenance record for the serving CBraMod artifact — resolved from the same
 * manifest + SHA-verified entry that `embedFoundationWindows` uses, so the
 * values reported to callers exactly match what was verified at load time.
 */
export interface FoundationProvenance {
  artifact_id: string;
  sha256: string;
  size: number;
  url: string;
  sample_rate_hz: number;
  window_samples: number;
  channels: number;
  embedding_dim: number;
  bandpass_hz: [number, number];
  output_pooling: "mean-tokens";
  normalization: string;
  runtime: string;
}

/** Read + parse the build-time artefact manifest from the filesystem (server-only). */
function loadManifest(): ArtefactManifest {
  const p = join(process.cwd(), "public", "models", "manifest.json");
  try {
    return JSON.parse(readFileSync(p, "utf-8")) as ArtefactManifest;
  } catch (e) {
    throw new FoundationUnavailableError(
      `cannot read artefact manifest at ${p}: ${(e as Error).message}`,
    );
  }
}

/** Build an `OrtRuntime` backed by the native onnxruntime-node CPU EP. */
export async function foundationRuntime(): Promise<OrtRuntime> {
  let ort: unknown;
  try {
    // Dynamic import isolates the native addon so it is only resolved when the
    // foundation route is actually invoked (server-side, Node.js runtime per T-003).
    ort = await import("onnxruntime-node");
  } catch (e) {
    throw new FoundationUnavailableError(`onnxruntime-node import failed: ${(e as Error).message}`);
  }
  const module = ort as {
    InferenceSession?: {
      create: (
        path: string | ArrayBuffer | Uint8Array,
        options?: Record<string, unknown>,
      ) => Promise<OrtSessionLike>;
    };
    Tensor?: new (type: "float32", data: Float32Array, dims: readonly number[]) => unknown;
  };
  if (!module.InferenceSession || !module.Tensor) {
    throw new FoundationUnavailableError("onnxruntime-node exposed no InferenceSession/Tensor");
  }
  // onnxruntime-node's Session exposes inputNames/outputNames/run/release
  // (verified against the CBraMod graph: inputs ["eeg"], outputs ["embedding"]),
  // so it conforms to OrtSessionLike directly — no adapter shim required.
  return {
    InferenceSession: {
      create: async (path, options) => module.InferenceSession!.create(path, options),
    },
    Tensor: module.Tensor as unknown as OrtRuntime["Tensor"],
  };
}

/** Cached warm ONNXAdapter session (reused across windows and requests). */
let cachedAdapter: ONNXAdapter | null = null;
/** Cached, SHA-verified artifact manifest entry (shared with foundationProvenance). */
let cachedEntry: ArtefactManifestEntry | null = null;

/**
 * Load + SHA-256 verify the CBraMod artifact, then build the cached ONNXAdapter.
 * Idempotent — the single loaded session is reused for every subsequent window.
 */
async function ensureAdapter(): Promise<ONNXAdapter> {
  if (cachedAdapter) return cachedAdapter;

  const manifest = loadManifest();
  const entry = manifest.models[FOUNDATION_ARTIFACT_ID];
  if (!entry) {
    throw new FoundationUnavailableError(
      `manifest has no entry for "${FOUNDATION_ARTIFACT_ID}" (Tier-2 not provisioned)`,
    );
  }
  if (entry.wasmCompatible !== false) {
    throw new FoundationUnavailableError(
      `manifest marks ${FOUNDATION_ARTIFACT_ID} wasmCompatible !== false; refusing Tier-2 route`,
    );
  }
  cachedEntry = entry;

  const artifactPath = join(process.cwd(), "public", entry.url.replace(/^\//, ""));
  if (!existsSync(artifactPath)) {
    throw new FoundationUnavailableError(`artifact not found at ${artifactPath}`);
  }
  const bytes = readFileSync(artifactPath);
  const u8 = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  // T-016 provenance: verify size + SHA-256 against the build-time manifest
  // BEFORE handing the bytes to the runtime. The digest (c128ccfd…) is the same
  // one Mission-11 validated at the file at rest.
  if (u8.byteLength !== entry.size) {
    throw new FoundationUnavailableError(
      `artifact size mismatch: expected ${entry.size}, got ${u8.byteLength}`,
    );
  }
  try {
    verifyArtefact(u8, entry.sha256);
  } catch (e) {
    throw new FoundationUnavailableError(`SHA-256 verification failed: ${(e as Error).message}`);
  }
  metrics.artifactVerificationTotal.inc({
    artifact: FOUNDATION_ARTIFACT_ID,
    outcome: "verified",
  });

  cachedAdapter = new ONNXAdapter({
    id: FOUNDATION_MODEL_ID,
    name: "CBraMod (server-native, 200-D)",
    version: "1.0.0",
    description:
      "CBraMod Conv+Transformer, native [1,19,1000]->[1,19,5,200], mean-tokens->200-D, L2. " +
      "wasmCompatible:false (DFT/ReduceL2). Server-only onnxruntime-node CPU EP.",
    task: "embedding",
    inputShape: {
      kind: "raw",
      channels: FOUNDATION_CHANNELS,
      samples: FOUNDATION_WINDOW_SAMPLES,
    },
    embeddingDim: FOUNDATION_EMBEDDING_DIM,
    outputPooling: "mean-tokens",
    artifact: u8,
    // No-op for bytes (enableVerification only acts on URL strings); we verify
    // above so the T-016 provenance gate is still enforced exactly once.
    enableVerification: true,
    executionProviders: ["cpu"],
    runtime: foundationRuntime,
    wasmCompatible: false,
    wasmBlockers: ["DFT", "ReduceL2"],
  });
  return cachedAdapter;
}

/**
 * Provenance record for the serving artifact. Synchronous — the manifest is read
 * from disk; by the time this is called after a successful embed, `cachedEntry`
 * is populated, so no extra filesystem read occurs in the hot path.
 */
export function foundationProvenance(): FoundationProvenance {
  const entry = cachedEntry ?? loadManifest().models[FOUNDATION_ARTIFACT_ID];
  if (!entry) {
    throw new FoundationUnavailableError(
      `manifest has no entry for "${FOUNDATION_ARTIFACT_ID}" (Tier-2 not provisioned)`,
    );
  }
  return {
    artifact_id: FOUNDATION_ARTIFACT_ID,
    sha256: entry.sha256,
    size: entry.size,
    url: entry.url,
    sample_rate_hz: FOUNDATION_SAMPLE_RATE_HZ,
    window_samples: FOUNDATION_WINDOW_SAMPLES,
    channels: FOUNDATION_CHANNELS,
    embedding_dim: FOUNDATION_EMBEDDING_DIM,
    bandpass_hz: [4, 38],
    output_pooling: "mean-tokens",
    normalization: "zscore per channel + mean-tokens L2",
    runtime: "onnxruntime-node cpu",
  };
}

/** Reset the cached adapter + provenance (test helper + SSR lifecycle hook). */
export function resetFoundationAdapter(): void {
  cachedAdapter?.unload().catch(() => {
    /* best-effort on teardown */
  });
  cachedAdapter = null;
  cachedEntry = null;
}

/**
 * Embed every 4 s window as an independent L2-normalized 200-D CBraMod vector.
 *
 * Each `window.data` is `[19][1000]` (19 CBraMod channels × 1000 samples @ 250 Hz),
 * exactly the input Mission-11 produced and the ONNXAdapter `buildTensor` contract.
 * No PCA / V2 fallback: any failure propagates to the caller, which returns 424/500.
 */
export async function embedFoundationWindows(windows: EEGWindow[]): Promise<EmbedResult[]> {
  if (windows.length === 0) {
    throw new FoundationUnavailableError("no preprocessing windows to embed");
  }
  const adapter = await ensureAdapter();
  await adapter.load(); // idempotent: no-op when the session is already cached

  const results: EmbedResult[] = [];
  for (const w of windows) {
    const t0 = performance.now();
    const out = await adapter.embed({ kind: "windows", windows: [w] } as ModelInput);
    metrics.foundationEmbedMs.observe({ model: FOUNDATION_MODEL_ID }, performance.now() - t0);
    // finalize(): validateEmbedding({expectedDim:200}) — throws on dim/NaN/zero —
    // then l2Normalize(). Produces an EmbedResult with fellBack=false.
    results.push(finalize(out, false, undefined, true, FOUNDATION_EMBEDDING_DIM));
  }
  return results;
}
