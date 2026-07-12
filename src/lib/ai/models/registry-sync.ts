/**
 * T-022 — Registry ↔ MLflow ↔ Storage three-way sync.
 *
 * The TS model registry is the runtime source of truth for which models
 * are available. This module syncs it with:
 *   1. **MLflow** (training-side metadata: params, metrics, run id)
 *   2. **Storage** (artefact-level: SHA-256 hash verification via T-009)
 *
 * The sync is pull-based: `syncRegistry()` fetches the latest run metadata
 * from MLflow, cross-references the artefact hash against the Storage
 * manifest (T-009), and returns a sync report. If a hash mismatch is
 * detected, the model is flagged and the embed facade should refuse to
 * route to it.
 */
import type { ArtefactManifest, ArtefactManifestEntry } from "../artefacts/hashed-artefact";
import { verifyArtefact } from "../artefacts/hashed-artefact";

/** MLflow run metadata (subset pulled via the REST API). */
export interface MLflowRunInfo {
  runId: string;
  experimentName: string;
  status: "running" | "finished" | "failed" | "killed" | "scheduled";
  params: Record<string, string>;
  metrics: Record<string, number>;
  /** Tags include `sha256_<artifact_name>` set by T-021's tracker. */
  tags: Record<string, string>;
  artefactUri?: string;
}

/** Result of syncing one model's registry ↔ MLflow ↔ Storage state. */
export interface SyncResult {
  modelId: string;
  mlflowRunId: string | null;
  storageHash: string | null;
  mlflowHash: string | null;
  verified: boolean;
  status: "synced" | "hash_mismatch" | "mlflow_unavailable" | "no_manifest";
  message: string;
}

/** Minimal MLflow REST client shape. */
export interface MLflowClient {
  getRun: (runId: string) => Promise<MLflowRunInfo | null>;
  searchRuns: (experimentName: string) => Promise<MLflowRunInfo[]>;
}

/**
 * Sync the registry for a single model by pulling its MLflow run metadata
 * and verifying the artefact hash against the Storage manifest.
 *
 * @param modelId       The registry model id (e.g. "braindecode-eegconformer-prod").
 * @param mlflow        MLflow REST client.
 * @param manifest      The Storage artefact manifest (from T-009).
 * @param artefactBytes The actual artefact bytes (fetched from Storage).
 * @returns A sync result indicating whether the three sources agree.
 */
export async function syncModelFromMLflow(
  modelId: string,
  mlflow: MLflowClient,
  manifest: ArtefactManifest,
  artefactBytes: Uint8Array,
): Promise<SyncResult> {
  const manifestEntry = findManifestEntry(manifest, modelId);
  if (!manifestEntry) {
    return {
      modelId,
      mlflowRunId: null,
      storageHash: null,
      mlflowHash: null,
      verified: false,
      status: "no_manifest",
      message: `No manifest entry for model ${modelId}`,
    };
  }

  // Verify the artefact bytes against the Storage manifest hash.
  try {
    verifyArtefact(artefactBytes, manifestEntry.sha256);
  } catch (e) {
    return {
      modelId,
      mlflowRunId: null,
      storageHash: manifestEntry.sha256,
      mlflowHash: null,
      verified: false,
      status: "hash_mismatch",
      message: `Storage hash verification failed: ${(e as Error).message}`,
    };
  }

  // Pull the latest MLflow run for this model.
  let runs: MLflowRunInfo[] = [];
  try {
    runs = await mlflow.searchRuns(modelId);
  } catch {
    return {
      modelId,
      mlflowRunId: null,
      storageHash: manifestEntry.sha256,
      mlflowHash: null,
      verified: false,
      status: "mlflow_unavailable",
      message: "MLflow tracking server unavailable",
    };
  }

  if (runs.length === 0) {
    // No MLflow run, but Storage hash verified — partially synced.
    return {
      modelId,
      mlflowRunId: null,
      storageHash: manifestEntry.sha256,
      mlflowHash: null,
      verified: true,
      status: "synced",
      message: "Storage hash verified; no MLflow run found",
    };
  }

  const latestRun = runs[0];
  const artefactFileName = manifestEntry.url.split("/").pop() ?? "";
  const mlflowHashTag = latestRun.tags[`sha256_${artefactFileName}`] ?? null;

  if (mlflowHashTag !== null && mlflowHashTag !== manifestEntry.sha256) {
    return {
      modelId,
      mlflowRunId: latestRun.runId,
      storageHash: manifestEntry.sha256,
      mlflowHash: mlflowHashTag,
      verified: false,
      status: "hash_mismatch",
      message: `MLflow tag hash ${mlflowHashTag.slice(0, 16)}… ≠ Storage hash ${manifestEntry.sha256.slice(0, 16)}…`,
    };
  }

  return {
    modelId,
    mlflowRunId: latestRun.runId,
    storageHash: manifestEntry.sha256,
    mlflowHash: mlflowHashTag,
    verified: true,
    status: "synced",
    message: `Registry ↔ MLflow ↔ Storage synced (run ${latestRun.runId})`,
  };
}

/**
 * Find the manifest entry for a model id. The manifest keys are filenames
 * (e.g. "eegconformer"), so we match by stripping known prefixes from the
 * model id (e.g. "braindecode-eegconformer-prod" → "eegconformer").
 */
function findManifestEntry(
  manifest: ArtefactManifest,
  modelId: string,
): ArtefactManifestEntry | null {
  // Direct match.
  if (manifest.models[modelId]) return manifest.models[modelId];

  // Heuristic: extract the architecture name from the model id.
  const archMatch = modelId.match(/(eegconformer|eegnetv4|shallowfbcspnet|deep4net)/i);
  if (archMatch) {
    const key = archMatch[1].toLowerCase();
    for (const id of Object.keys(manifest.models)) {
      if (id.includes(key)) return manifest.models[id];
    }
  }

  return null;
}
