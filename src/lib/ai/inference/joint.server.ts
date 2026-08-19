/**
 * M25 — Joint 264-D embedding fusion (server-only, `.server.ts` suffix).
 *
 * Fuses CBraMod-200 (onnxruntime-node), EEGConformer V2-32 (onnxruntime-node),
 * and PCA-32 (pure JS) into a single 264-D embedding using the fixed block
 * weights learned in M18:  [CBraMod×0.62 ⊕ V2×0.16 ⊕ PCA×0.22].
 *
 * WHY THIS IS SERVER-ONLY: both CBraMod and the fused joint path require
 * onnxruntime-node (CBraMod) or the same native runtime (V2 for parity). The
 * PCA component is pure JS, but the joint vector is only meaningful when all
 * three blocks are computed with the same provenance. The `.server.ts` suffix
 * ensures Vinxi / TanStack Start keeps this out of the browser bundle, and
 * `onnxruntime-node` is imported DYNAMICALLY inside `foundationRuntime()` (shared
 * with foundation.server.ts) so non-node clients never load the native addon.
 *
 * PIPELINE:
 *   windows19 (19-ch, CBraMod)  → embedFoundationWindows → 200-D, L2-norm
 *   windows22 (22-ch, V2)       → embedV2Windows         →  32-D, L2-norm
 *   windows22 (22-ch, PCA)      → embedPCAWindows        →  32-D, L2-norm
 *
 *   fuse: concat([CBraMod-200, V2-32, PCA-32])  → 264-D
 *         per-block L2-normalise → block-weight scaling [0.62, 0.16, 0.22]
 *         → final L2-normalise
 *
 * M18 validated this fusion: R@5=0.7856 (p=4.5e-9), beating PCA-32 (0.740),
 * CBraMod-200 (0.528), and V2-32 (0.216). Block weights are fixed and were
 * stable across all 50 LOSO folds.
 *
 * NO FALLBACK BY DESIGN: like foundation.server.ts, this module throws on any
 * failure (runtime, artifact, per-window). It never degrades to a single model
 * or PCA-only path. The HTTP layer maps runtime/artifact unavailability to 424.
 */
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

import { ONNXAdapter } from "../adapters/onnx-adapter";
import { finalize, type EmbedResult } from "../embeddings";
import type { EmbeddingOutput } from "../types";
import { l2Normalize, validateEmbedding } from "../validation";
import {
  verifyArtefact,
  type ArtefactManifest,
  type ArtefactManifestEntry,
} from "../artefacts/hashed-artefact";
import type { ModelInput } from "../types";
import type { EEGWindow } from "../../eeg/types";
import { PROD_CHANNEL_COUNT } from "../../eeg/channels";
import { bandPowerFeatures } from "../../embeddings/features";
import { fitPCA, transformPCA, type PCAModel } from "../../embeddings/pca";
import { log } from "../../logging";
import { metrics } from "../../metrics";
import {
  embedFoundationWindows,
  FOUNDATION_MODEL_ID,
  FOUNDATION_EMBEDDING_DIM,
  foundationRuntime,
  FoundationUnavailableError,
} from "./foundation.server";

// ──────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────

/** Production model id for the fused joint-264 embedding. */
export const JOINT_MODEL_ID = "onnx-cbramod-joint-264";
/** Joint embedding dimension: CBraMod-200 ⊕ V2-32 ⊕ PCA-32. */
export const JOINT_EMBEDDING_DIM = 264;

/**
 * Fixed block weights (M18 — stable across all 50 LOSO folds):
 *   CBraMod = 0.62, V2 = 0.16, PCA = 0.22  (sum = 1.00)
 */
export const JOINT_BLOCK_WEIGHTS = {
  cbramod: 0.62,
  v2: 0.16,
  pca: 0.22,
} as const;

/** Component dimensions in the joint vector. */
export const JOINT_COMPONENT_DIMS = {
  cbramod: 200,
  v2: 32,
  pca: 32,
} as const;

// ──────────────────────────────────────────────────────────────────────────
// EEGPT constants (read-only artifact: SHA a92daf44…)
// ──────────────────────────────────────────────────────────────────────────

/** Manifest artifact id for the INT8-quantised EEGPT model. */
export const EEGPT_ARTIFACT_ID = "eegpt-encoder-int8";
/** EEGPT embedding dimension after mean-tokens pooling over the [1, 31, 2048] output. */
export const EEGPT_EMBEDDING_DIM = 2048;
/** EEGPT expects the standard 62-channel 10-20 montage at 250 Hz. */
export const EEGPT_CHANNELS = 62;
export const EEGPT_SAMPLE_RATE_HZ = 250;
export const EEGPT_WINDOW_SAMPLES = 1000;
/** SHA-256 of public/models/eegpt-encoder-int8.onnx (verified in M26/M27). */
export const EEGPT_SHA256 = "a92daf44ab8cb10e4c258c853b2d4668232acc6e09606fa2e18d052c4b914f36";

// ──────────────────────────────────────────────────────────────────────────
// Joint-2312 constants (M27 — 4-block fusion: CBraMod + V2 + PCA + EEGPT)
// ──────────────────────────────────────────────────────────────────────────

/** Production model id for the fused joint-2312 embedding (4-block). */
export const JOINT_2312_MODEL_ID = "onnx-cbramod-joint-2312";
/** Joint-2312 embedding dimension: CBraMod-200 ⊕ V2-32 ⊕ PCA-32 ⊕ EEGPT-2048. */
export const JOINT_2312_EMBEDDING_DIM = 2312;

/**
 * M27 learned 4-block weights (stable across all 50 LOSO folds, CV < 2%):
 *   CBraMod = 0.3062, V2 = 0.1434, PCA = 0.1519, EEGPT = 0.3985  (sum ≈ 1.00)
 *
 * EEGPT receives the largest weight (0.399), reflecting its highest individual
 * retrieval quality (R@5=0.8118) and complementary signal to CBraMod/V2/PCA.
 * These are FIXED for production — the M27 evaluation proved per-fold learning
 * does not change the weights materially (weight CV < 0.5%).
 */
export const JOINT_2312_BLOCK_WEIGHTS = {
  cbramod: 0.3062,
  v2: 0.1434,
  pca: 0.1519,
  eegpt: 0.3985,
} as const;

/** Component dimensions in the joint-2312 vector. */
export const JOINT_2312_COMPONENT_DIMS = {
  cbramod: 200,
  v2: 32,
  pca: 32,
  eegpt: 2048,
} as const;

// V2 (EEGConformer-prod-v2) artifact constants — mirrors the manifest entry and
// the registry's DEFAULT_PREFERRED model. The SHA and size MUST match the
// read-only artifact at public/models/eegconformer_finetuned.onnx.
export const V2_ARTIFACT_ID = "eegconformer_finetuned";
export const V2_EMBEDDING_DIM = 32;
export const V2_CHANNELS = 22;
export const V2_SAMPLE_RATE_HZ = 250;
export const V2_WINDOW_SAMPLES = 1000;
export const V2_SHA256 = "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931";

// PCA (band-power) constants — mirrors the Tier-1 PCA contract.
export const PCA_BANDS_PER_CHANNEL = 5; // δ θ α β γ
export const PCA_INPUT_FEATURES = PROD_CHANNEL_COUNT * PCA_BANDS_PER_CHANNEL; // 110
export const PCA_OUTPUT_DIM = 32;

/** Raised when the V2 artifact or runtime is unavailable. */
export class V2UnavailableError extends FoundationUnavailableError {
  constructor(reason: string) {
    super(`V2: ${reason}`);
    this.name = "V2UnavailableError";
  }
}

/** Raised when the EEGPT artifact or runtime is unavailable. */
export class EEGPTUnavailableError extends FoundationUnavailableError {
  constructor(reason: string) {
    super(`EEGPT: ${reason}`);
    this.name = "EEGPTUnavailableError";
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Provenance
// ──────────────────────────────────────────────────────────────────────────

export interface JointProvenance {
  artifact_ids: { cbramod: string; v2: string };
  cbramod_sha256: string;
  v2_sha256: string;
  v2_size: number;
  embedding_dim: number;
  component_dims: typeof JOINT_COMPONENT_DIMS;
  block_weights: typeof JOINT_BLOCK_WEIGHTS;
  runtime: string;
}

// ──────────────────────────────────────────────────────────────────────────
// Manifest loading (server-only; reads from the build-time manifest)
// ──────────────────────────────────────────────────────────────────────────

/** Read + parse the build-time artefact manifest from the filesystem. */
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

// ──────────────────────────────────────────────────────────────────────────
// Cached V2 adapter (onnxruntime-node, SHA-verified)
// ──────────────────────────────────────────────────────────────────────────

/** Cached warm ONNXAdapter session for V2 (reused across windows and requests). */
let cachedV2Adapter: ONNXAdapter | null = null;
/** Cached, SHA-verified V2 manifest entry (shared with jointProvenance). */
let cachedV2Entry: ArtefactManifestEntry | null = null;

/**
 * Load + SHA-256 verify the V2 artifact, then build the cached ONNXAdapter.
 * Idempotent — the single loaded session is reused for every subsequent window.
 *
 * Mirrors foundation.server.ts `ensureAdapter()` but for V2 instead of CBraMod.
 * The V2 artifact is wasmCompatible:true (no DFT/ReduceL2), but we still run it on
 * the server via onnxruntime-node CPU EP for pipeline parity with CBraMod.
 */
async function ensureV2Adapter(): Promise<ONNXAdapter> {
  if (cachedV2Adapter) return cachedV2Adapter;

  const manifest = loadManifest();
  const entry = manifest.models[V2_ARTIFACT_ID];
  if (!entry) {
    throw new V2UnavailableError(
      `manifest has no entry for "${V2_ARTIFACT_ID}" (V2 not provisioned)`,
    );
  }
  cachedV2Entry = entry;

  const artifactPath = join(process.cwd(), "public", entry.url.replace(/^\//, ""));
  if (!existsSync(artifactPath)) {
    throw new V2UnavailableError(`artifact not found at ${artifactPath}`);
  }
  const bytes = readFileSync(artifactPath);
  const u8 = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  // T-016 provenance: verify size + SHA-256 against the build-time manifest
  // BEFORE handing the bytes to the runtime. The digest (18644de1…) is the same
  // one M18 validated at the file at rest.
  if (u8.byteLength !== entry.size) {
    throw new V2UnavailableError(
      `V2 artifact size mismatch: expected ${entry.size}, got ${u8.byteLength}`,
    );
  }
  try {
    verifyArtefact(u8, entry.sha256);
  } catch (e) {
    throw new V2UnavailableError(`V2 SHA-256 verification failed: ${(e as Error).message}`);
  }
  metrics.artifactVerificationTotal.inc({
    artifact: V2_ARTIFACT_ID,
    outcome: "verified",
  });

  cachedV2Adapter = new ONNXAdapter({
    id: "onnx-v2-joint-32d",
    name: "EEGConformer V2 (server-native, 32-D)",
    version: "1.0.0",
    description:
      "EEGConformer v2 fine-tuned on PhysioNet EEGMMIDB (T-031). " +
      "Server-side onnxruntime-node CPU EP. Input [1,22,1000]→[1,32]. " +
      `SHA ${entry.sha256.slice(0, 16)}… (read-only, M23 GA default).`,
    task: "embedding",
    inputShape: {
      kind: "raw",
      channels: V2_CHANNELS,
      samples: V2_WINDOW_SAMPLES,
    },
    embeddingDim: V2_EMBEDDING_DIM,
    outputPooling: "none", // V2 output is already flat [1, 32]
    artifact: u8,
    // No-op for bytes (enableVerification only acts on URL strings); we verify
    // above so the T-016 provenance gate is still enforced exactly once.
    enableVerification: true,
    executionProviders: ["cpu"],
    runtime: foundationRuntime,
    wasmCompatible: true,
  });
  return cachedV2Adapter;
}

/** Cached warm ONNXAdapter session for EEGPT (reused across windows and requests). */
let cachedEEGPTAdapter: ONNXAdapter | null = null;
/** Cached, SHA-verified EEGPT manifest entry (shared with joint2312Provenance). */
let cachedEEGPTEntry: ArtefactManifestEntry | null = null;

/**
 * Load + SHA-256 verify the EEGPT artifact, then build the cached ONNXAdapter.
 * Idempotent — the single loaded session is reused for every subsequent window.
 *
 * Mirrors `ensureV2Adapter()` but for EEGPT instead of V2. EEGPT is wasmCompatible
 * (no DFT/ReduceL2 blockers), but we run it on the server via onnxruntime-node CPU EP
 * for pipeline parity with CBraMod and V2 in the joint-2312 path.
 */
async function ensureEEGPTAdapter(): Promise<ONNXAdapter> {
  if (cachedEEGPTAdapter) return cachedEEGPTAdapter;

  const manifest = loadManifest();
  const entry = manifest.models[EEGPT_ARTIFACT_ID];
  if (!entry) {
    throw new EEGPTUnavailableError(
      `manifest has no entry for "${EEGPT_ARTIFACT_ID}" (EEGPT not provisioned)`,
    );
  }
  cachedEEGPTEntry = entry;

  const artifactPath = join(process.cwd(), "public", entry.url.replace(/^\//, ""));
  if (!existsSync(artifactPath)) {
    throw new EEGPTUnavailableError(`artifact not found at ${artifactPath}`);
  }
  const bytes = readFileSync(artifactPath);
  const u8 = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  // T-016 provenance: verify size + SHA-256 against the build-time manifest
  // BEFORE handing the bytes to the runtime. The digest (a92daf44…) is the same
  // one M26/M27 validated at the file at rest.
  if (u8.byteLength !== entry.size) {
    throw new EEGPTUnavailableError(
      `EEGPT artifact size mismatch: expected ${entry.size}, got ${u8.byteLength}`,
    );
  }
  try {
    verifyArtefact(u8, entry.sha256);
  } catch (e) {
    throw new EEGPTUnavailableError(`EEGPT SHA-256 verification failed: ${(e as Error).message}`);
  }
  metrics.artifactVerificationTotal.inc({
    artifact: EEGPT_ARTIFACT_ID,
    outcome: "verified",
  });

  cachedEEGPTAdapter = new ONNXAdapter({
    id: JOINT_2312_MODEL_ID + "-eegpt",
    name: "EEGPT (server-native, 2048-D)",
    version: "1.0.0",
    description:
      "EEGPT INT8-viT, native [1,62,1000]->[1,31,2048], mean-tokens->2048-D, L2. " +
      "wasmCompatible (no DFT/ReduceL2). Server-side onnxruntime-node CPU EP. " +
      `SHA ${entry.sha256.slice(0, 16)}… (read-only, M26/M27 validated).`,
    task: "embedding",
    inputShape: {
      kind: "raw",
      channels: EEGPT_CHANNELS,
      samples: EEGPT_WINDOW_SAMPLES,
    },
    embeddingDim: EEGPT_EMBEDDING_DIM,
    outputPooling: "mean-tokens",
    artifact: u8,
    enableVerification: true,
    executionProviders: ["cpu"],
    runtime: foundationRuntime,
    wasmCompatible: true,
  });
  return cachedEEGPTAdapter;
}

// ──────────────────────────────────────────────────────────────────────────
// PCA (pure JS) — band-power features → PCA(32) → L2-normalise
// ──────────────────────────────────────────────────────────────────────────

/** Standardise features: mean=0, std=1 per column (mirrors sklearn StandardScaler). */
function standardizeFeatures(X: number[][]): {
  standardized: number[][];
  mean: number[];
  std: number[];
} {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  if (n === 0 || d === 0) throw new Error("PCA: empty feature matrix");

  const mean = new Array<number>(d).fill(0);
  for (const row of X) for (let j = 0; j < d; j++) mean[j] += row[j];
  for (let j = 0; j < d; j++) mean[j] /= n;

  const std = new Array<number>(d).fill(0);
  for (const row of X) for (let j = 0; j < d; j++) std[j] += (row[j] - mean[j]) ** 2;
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / n) || 1;

  const standardized = X.map((row) => row.map((v, j) => (v - mean[j]) / std[j]));
  return { standardized, mean, std };
}

/**
 * Compute per-window PCA-32 band-power embeddings (L2-normalised).
 *
 * Mirrors M18's PCA computation:
 *   1. bandPowerFeatures per window → [C × 5 = 110] feature vector
 *   2. Standardise (StandardScaler: mean=0, std=1 per feature)
 *   3. Fit PCA(32) via power iteration on the covariance matrix
 *   4. Transform each window → 32-D
 *   5. L2-normalise
 *
 * Unlike the `embed()` facade (which mean-pools across windows before PCA),
 * this preserves per-window structure so each EEGWindow maps to one 32-D vector.
 */
export function embedPCAWindows(windows: EEGWindow[]): number[][] {
  if (windows.length === 0) {
    throw new FoundationUnavailableError("no PCA windows to embed");
  }

  // 1. Band-power features per window [n_windows × 110]
  const features = windows.map(bandPowerFeatures);

  // 2. Standardise
  const { standardized } = standardizeFeatures(features);

  // 3. Fit PCA(32) on the full batch (standard production approach — the
  //    per-fold train-only fitting was an M18 evaluation detail, not a
  //    production requirement for a single inference request).
  const pca: PCAModel = fitPCA(standardized, PCA_OUTPUT_DIM, 60);

  // 4+5. Transform each window → 32-D → L2-normalise
  return standardized.map((f) => {
    const proj = transformPCA(pca, f);
    validateEmbedding(proj, { expectedDim: PCA_OUTPUT_DIM });
    return l2Normalize(proj);
  });
}

// ──────────────────────────────────────────────────────────────────────────
// V2 embedding (onnxruntime-node, SHA-verified) — per window
// ──────────────────────────────────────────────────────────────────────────

/**
 * Embed 22-channel windows through the V2 ONNX model (onnxruntime-node CPU EP).
 *
 * Each `window.data` is `[22][1000]` (22 prod channels × 1000 samples @ 250 Hz),
 * exactly the V2 model's input contract. Returns L2-normalised 32-D vectors.
 */
export async function embedV2Windows(windows: EEGWindow[]): Promise<number[][]> {
  if (windows.length === 0) {
    throw new V2UnavailableError("no V2 windows to embed");
  }
  const adapter = await ensureV2Adapter();
  await adapter.load(); // idempotent: no-op when the session is already cached

  const results: number[][] = [];
  for (const w of windows) {
    const t0 = performance.now();
    const out = await adapter.embed({ kind: "windows", windows: [w] } as ModelInput);
    const durationMs = +(performance.now() - t0).toFixed(2);
    log("debug", "v2.embed.window", { dim: out.dim, durationMs });
    // Validate (dim=32, no NaN, no zero) then L2-normalise.
    validateEmbedding(out.vector, { expectedDim: V2_EMBEDDING_DIM });
    results.push(l2Normalize(out.vector));
  }
  return results;
}

/**
 * Embed 62-channel EEGPT windows (onnxruntime-node CPU EP, SHA-verified).
 *
 * Each `window.data` is `[62][1000]` (62 EEGPT channels × 1000 samples @ 250 Hz),
 * exactly the EEGPT model's input contract. The ONNXAdapter's `applyOutputPooling`
 * mean-token-pools the `[1, 31, 2048]` raw output to 2048-D. Returns
 * L2-normalised 2048-D vectors.
 */
export async function embedEEGPTWindows(windows: EEGWindow[]): Promise<number[][]> {
  if (windows.length === 0) {
    throw new EEGPTUnavailableError("no EEGPT windows to embed");
  }
  const adapter = await ensureEEGPTAdapter();
  await adapter.load(); // idempotent: no-op when the session is already cached

  const results: number[][] = [];
  for (const w of windows) {
    const t0 = performance.now();
    const out = await adapter.embed({ kind: "windows", windows: [w] } as ModelInput);
    const durationMs = +(performance.now() - t0).toFixed(2);
    log("debug", "eegpt.embed.window", { dim: out.dim, durationMs });
    // Validate (dim=2048, no NaN, no zero) then L2-normalise.
    validateEmbedding(out.vector, { expectedDim: EEGPT_EMBEDDING_DIM });
    results.push(l2Normalize(out.vector));
  }
  return results;
}

// ──────────────────────────────────────────────────────────────────────────
// Joint fusion
// ──────────────────────────────────────────────────────────────────────────

/**
 * Fuse CBraMod-200, V2-32, and PCA-32 into a single 264-D embedding.
 *
 * Mirrors M18's `apply_block_weights`:
 *   1. L2-normalise each block independently
 *   2. Scale each block by its learned weight [0.62, 0.16, 0.22]
 *   3. Concatenate → 264-D
 *   4. L2-normalise the full vector
 */
export function fuseJointEmbedding(
  cbVector: number[],
  v2Vector: number[],
  pcaVector: number[],
): number[] {
  // Validate block dimensions
  if (cbVector.length !== FOUNDATION_EMBEDDING_DIM) {
    throw new Error(
      `Joint fusion: CBraMod vector dim ${cbVector.length} != ${FOUNDATION_EMBEDDING_DIM}`,
    );
  }
  if (v2Vector.length !== V2_EMBEDDING_DIM) {
    throw new Error(`Joint fusion: V2 vector dim ${v2Vector.length} != ${V2_EMBEDDING_DIM}`);
  }
  if (pcaVector.length !== PCA_OUTPUT_DIM) {
    throw new Error(`Joint fusion: PCA vector dim ${pcaVector.length} != ${PCA_OUTPUT_DIM}`);
  }

  // 1. L2-normalise each block (idempotent if already normalised)
  const cbN = l2Normalize(cbVector);
  const v2N = l2Normalize(v2Vector);
  const pcaN = l2Normalize(pcaVector);

  // 2+3. Scale by block weights and concatenate
  const w = JOINT_BLOCK_WEIGHTS;
  const result = new Array<number>(JOINT_EMBEDDING_DIM);
  let i = 0;
  for (let j = 0; j < cbN.length; j++) result[i++] = w.cbramod * cbN[j];
  for (let j = 0; j < v2N.length; j++) result[i++] = w.v2 * v2N[j];
  for (let j = 0; j < pcaN.length; j++) result[i++] = w.pca * pcaN[j];

  // 4. Final L2-normalise
  return l2Normalize(result);
}

/**
 * Embed aligned 19-channel (CBraMod) and 22-channel (V2/PCA) windows into
 * 264-D joint embeddings.
 *
 * Each `windows19[i]` and `windows22[i]` must correspond to the same time
 * segment (same window count, same sample rate, same start/end). The function
 * runs all three embedders and fuses per-window using M18's fixed block weights.
 */
export async function embedJointWindows(
  windows19: EEGWindow[],
  windows22: EEGWindow[],
): Promise<EmbedResult[]> {
  if (windows19.length !== windows22.length) {
    throw new FoundationUnavailableError(
      `window count mismatch: CBraMod has ${windows19.length} windows, ` +
        `V2/PCA has ${windows22.length}`,
    );
  }
  if (windows19.length === 0) {
    throw new FoundationUnavailableError("no windows to embed");
  }

  const t0 = performance.now();

  // CBraMod-200 (19-channel, server-native onnxruntime-node, SHA-verified)
  const cbResults = await embedFoundationWindows(windows19);

  // V2-32 (22-channel, server-native onnxruntime-node, SHA-verified)
  const v2Vectors = await embedV2Windows(windows22);

  // PCA-32 (22-channel band-power features, pure JS)
  const pcaVectors = embedPCAWindows(windows22);

  // Fuse each window: [CBraMod-200, V2-32, PCA-32] → 264-D
  const results: EmbedResult[] = [];
  for (let i = 0; i < cbResults.length; i++) {
    const fused = fuseJointEmbedding(cbResults[i].vector, v2Vectors[i], pcaVectors[i]);
    const out: EmbeddingOutput = {
      vector: fused,
      dim: fused.length,
      modelId: JOINT_MODEL_ID,
      durationMs: 0,
    };
    // finalize: validateEmbedding({expectedDim:264}) + L2-normalise (idempotent
    // since fuseJointEmbedding already normalised). fellBack=false by design.
    results.push(finalize(out, false, undefined, true, JOINT_EMBEDDING_DIM));
  }

  const totalMs = +(performance.now() - t0).toFixed(2);
  log("info", "joint.embed.done", {
    modelId: JOINT_MODEL_ID,
    dim: JOINT_EMBEDDING_DIM,
    windows: results.length,
    totalMs,
  });

  return results;
}

// ──────────────────────────────────────────────────────────────────────────
// Joint-2312 fusion (M27: 4-block with EEGPT-2048)
// ──────────────────────────────────────────────────────────────────────────

/**
 * Fuse CBraMod-200, V2-32, PCA-32, and EEGPT-2048 into a single 2312-D embedding.
 *
 * Mirrors M18's `apply_block_weights` / M25's `fuseJointEmbedding`, extended to 4
 * blocks with the M27-learned weights:
 *   1. L2-normalise each block independently
 *   2. Scale each block by its learned weight [0.3062, 0.1434, 0.1519, 0.3985]
 *   3. Concatenate → 2312-D
 *   4. L2-normalise the full vector
 */
export function fuseJoint2312Embedding(
  cbVector: number[],
  v2Vector: number[],
  pcaVector: number[],
  eegptVector: number[],
): number[] {
  // Validate block dimensions
  if (cbVector.length !== FOUNDATION_EMBEDDING_DIM) {
    throw new Error(
      `Joint-2312 fusion: CBraMod vector dim ${cbVector.length} != ${FOUNDATION_EMBEDDING_DIM}`,
    );
  }
  if (v2Vector.length !== V2_EMBEDDING_DIM) {
    throw new Error(`Joint-2312 fusion: V2 vector dim ${v2Vector.length} != ${V2_EMBEDDING_DIM}`);
  }
  if (pcaVector.length !== PCA_OUTPUT_DIM) {
    throw new Error(`Joint-2312 fusion: PCA vector dim ${pcaVector.length} != ${PCA_OUTPUT_DIM}`);
  }
  if (eegptVector.length !== EEGPT_EMBEDDING_DIM) {
    throw new Error(
      `Joint-2312 fusion: EEGPT vector dim ${eegptVector.length} != ${EEGPT_EMBEDDING_DIM}`,
    );
  }

  // 1. L2-normalise each block (idempotent if already normalised)
  const cbN = l2Normalize(cbVector);
  const v2N = l2Normalize(v2Vector);
  const pcaN = l2Normalize(pcaVector);
  const eegptN = l2Normalize(eegptVector);

  // 2+3. Scale by block weights and concatenate
  const w = JOINT_2312_BLOCK_WEIGHTS;
  const result = new Array<number>(JOINT_2312_EMBEDDING_DIM);
  let i = 0;
  for (let j = 0; j < cbN.length; j++) result[i++] = w.cbramod * cbN[j];
  for (let j = 0; j < v2N.length; j++) result[i++] = w.v2 * v2N[j];
  for (let j = 0; j < pcaN.length; j++) result[i++] = w.pca * pcaN[j];
  for (let j = 0; j < eegptN.length; j++) result[i++] = w.eegpt * eegptN[j];

  // 4. Final L2-normalise
  return l2Normalize(result);
}

/**
 * Embed aligned 19-channel (CBraMod), 22-channel (V2/PCA), and 62-channel (EEGPT)
 * windows into 2312-D joint embeddings.
 *
 * Each `windows19[i]`, `windows22[i]`, and `windows62[i]` must correspond to the
 * same time segment (same window count, same sample rate, same start/end). The
 * function runs all four embedders and fuses per-window using M27's learned block
 * weights.
 */
export async function embedJoint2312Windows(
  windows19: EEGWindow[],
  windows22: EEGWindow[],
  windows62: EEGWindow[],
): Promise<EmbedResult[]> {
  if (windows19.length !== windows22.length || windows19.length !== windows62.length) {
    throw new FoundationUnavailableError(
      `window count mismatch: CBraMod has ${windows19.length} windows, ` +
        `V2/PCA has ${windows22.length}, EEGPT has ${windows62.length}`,
    );
  }
  if (windows19.length === 0) {
    throw new FoundationUnavailableError("no windows to embed");
  }

  const t0 = performance.now();

  // CBraMod-200 (19-channel, server-native onnxruntime-node, SHA-verified)
  const cbResults = await embedFoundationWindows(windows19);

  // V2-32 (22-channel, server-native onnxruntime-node, SHA-verified)
  const v2Vectors = await embedV2Windows(windows22);

  // PCA-32 (22-channel band-power features, pure JS)
  const pcaVectors = embedPCAWindows(windows22);

  // EEGPT-2048 (62-channel, server-native onnxruntime-node, SHA-verified)
  const eegptVectors = await embedEEGPTWindows(windows62);

  // Fuse each window: [CBraMod-200, V2-32, PCA-32, EEGPT-2048] → 2312-D
  const results: EmbedResult[] = [];
  for (let i = 0; i < cbResults.length; i++) {
    const fused = fuseJoint2312Embedding(
      cbResults[i].vector,
      v2Vectors[i],
      pcaVectors[i],
      eegptVectors[i],
    );
    const out: EmbeddingOutput = {
      vector: fused,
      dim: fused.length,
      modelId: JOINT_2312_MODEL_ID,
      durationMs: 0,
    };
    // finalize: validateEmbedding({expectedDim:2312}) + L2-normalise (idempotent
    // since fuseJoint2312Embedding already normalised). fellBack=false by design.
    results.push(finalize(out, false, undefined, true, JOINT_2312_EMBEDDING_DIM));
  }

  const totalMs = +(performance.now() - t0).toFixed(2);
  log("info", "joint2312.embed.done", {
    modelId: JOINT_2312_MODEL_ID,
    dim: JOINT_2312_EMBEDDING_DIM,
    windows: results.length,
    totalMs,
  });

  return results;
}

// ──────────────────────────────────────────────────────────────────────────
// Provenance
// ──────────────────────────────────────────────────────────────────────────

/**
 * Provenance record for the serving joint embedding. Resolved from the same
 * manifest that `ensureV2Adapter` and `embedFoundationWindows` use, so the
 * artifact digests reported to callers exactly match what was verified at load.
 */
export function jointProvenance(): JointProvenance {
  const manifest = loadManifest();
  const cbramodEntry = manifest.models["cbramod-encoder"];
  const v2Entry = cachedV2Entry ?? manifest.models[V2_ARTIFACT_ID];
  if (!cbramodEntry) {
    throw new FoundationUnavailableError(
      'manifest has no entry for "cbramod-encoder" (CBraMod not provisioned)',
    );
  }
  if (!v2Entry) {
    throw new FoundationUnavailableError(
      `manifest has no entry for "${V2_ARTIFACT_ID}" (V2 not provisioned)`,
    );
  }
  return {
    artifact_ids: { cbramod: "cbramod-encoder", v2: V2_ARTIFACT_ID },
    cbramod_sha256: cbramodEntry.sha256,
    v2_sha256: v2Entry.sha256,
    v2_size: v2Entry.size,
    embedding_dim: JOINT_EMBEDDING_DIM,
    component_dims: JOINT_COMPONENT_DIMS,
    block_weights: JOINT_BLOCK_WEIGHTS,
    runtime: "onnxruntime-node cpu + js",
  };
}

/** Provenance record for the serving joint-2312 (4-block) embedding. */
export interface Joint2312Provenance {
  artifact_ids: { cbramod: string; v2: string; eegpt: string };
  cbramod_sha256: string;
  v2_sha256: string;
  v2_size: number;
  eegpt_sha256: string;
  eegpt_size: number;
  embedding_dim: number;
  component_dims: typeof JOINT_2312_COMPONENT_DIMS;
  block_weights: typeof JOINT_2312_BLOCK_WEIGHTS;
  runtime: string;
}

/**
 * Provenance record for the serving joint-2312 embedding. Resolved from the same
 * manifest that `ensureEEGPTAdapter` and `embedFoundationWindows` use, so the
 * artifact digests reported to callers exactly match what was verified at load.
 */
export function joint2312Provenance(): Joint2312Provenance {
  const manifest = loadManifest();
  const cbramodEntry = manifest.models["cbramod-encoder"];
  const v2Entry = cachedV2Entry ?? manifest.models[V2_ARTIFACT_ID];
  const eegptEntry = cachedEEGPTEntry ?? manifest.models[EEGPT_ARTIFACT_ID];
  if (!cbramodEntry) {
    throw new FoundationUnavailableError(
      'manifest has no entry for "cbramod-encoder" (CBraMod not provisioned)',
    );
  }
  if (!v2Entry) {
    throw new FoundationUnavailableError(
      `manifest has no entry for "${V2_ARTIFACT_ID}" (V2 not provisioned)`,
    );
  }
  if (!eegptEntry) {
    throw new EEGPTUnavailableError(
      `manifest has no entry for "${EEGPT_ARTIFACT_ID}" (EEGPT not provisioned)`,
    );
  }
  return {
    artifact_ids: { cbramod: "cbramod-encoder", v2: V2_ARTIFACT_ID, eegpt: EEGPT_ARTIFACT_ID },
    cbramod_sha256: cbramodEntry.sha256,
    v2_sha256: v2Entry.sha256,
    v2_size: v2Entry.size,
    eegpt_sha256: eegptEntry.sha256,
    eegpt_size: eegptEntry.size,
    embedding_dim: JOINT_2312_EMBEDDING_DIM,
    component_dims: JOINT_2312_COMPONENT_DIMS,
    block_weights: JOINT_2312_BLOCK_WEIGHTS,
    runtime: "onnxruntime-node cpu + js",
  };
}

/** Reset all cached adapters + provenance entries (test helper + SSR lifecycle hook). */
export function resetJointAdapter(): void {
  cachedV2Adapter?.unload().catch(() => {
    /* best-effort on teardown */
  });
  cachedV2Adapter = null;
  cachedV2Entry = null;
  cachedEEGPTAdapter?.unload().catch(() => {
    /* best-effort on teardown */
  });
  cachedEEGPTAdapter = null;
  cachedEEGPTEntry = null;
}
