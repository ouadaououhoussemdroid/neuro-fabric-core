/**
 * M31 Shared Service Layer — ServiceProvenance (server-side).
 *
 * Every downstream Tier-1 service result carries a provenance record that
 * identifies the exact embedding artifacts, task head, dataset, and validation
 * metrics that produced it.
 *
 * This module is server-only (`.server.ts` suffix) because it reads the
 * runtime SHA-256 digests from `joint2312Provenance()` — which loads the
 * build-time artifact manifest and verifies SHA-256 digests at computation
 * time. The client-side code never needs the raw digests; it receives them
 * as part of the `ServiceProvenance` JSON in the API response.
 *
 * All three Tier-1 services (Subject Identity, Cognitive State, Anomaly
 * Detection) produce a ServiceProvenance on every prediction.
 */
import {
  JOINT_2312_MODEL_ID,
  JOINT_2312_EMBEDDING_DIM,
  JOINT_2312_BLOCK_WEIGHTS,
  JOINT_2312_COMPONENT_DIMS,
  joint2312Provenance,
} from "../inference/joint.server";

/** SHA-256 digests of the 4 Joint-2312 embedding artifacts. */
export interface EmbeddingArtifactSHAs {
  cbramod: string;
  v2: string;
  pca: string;
  eegpt: string;
}

/**
 * Provenance record attached to every Tier-1 downstream service result.
 *
 * Reuses the artifact SHAs and block weights from joint.server.ts so that
 * the provenance reported to callers exactly matches what was verified at
 * embedding computation time.
 */
export interface ServiceProvenance {
  /** The downstream service id (e.g. "subject-identity", "cognitive-intelligence", "anomaly-detection"). */
  service: string;
  /** Service version (semver tag of the service code, not a model version). */
  service_version: string;
  /** The upstream embedding model that produced the backbone embedding. */
  embedding_model: string;
  /** The upstream embedding dimension. */
  embedding_dim: number;
  /** SHA-256 digests of the 4 embedding artifacts. */
  artifact_shas: EmbeddingArtifactSHAs;
  /** The task-head model id (e.g. "subject-identity-mahalanobis-v1"). */
  task_head_id: string;
  /** Task-head version (git SHA or semver). */
  task_head_version?: string;
  /** Optional SHA-256 of the task-head artifact. */
  task_head_sha256?: string;
  /** The dataset used to train/evaluate the task head. */
  task_head_dataset?: string;
  /** Evaluation metrics from head validation (e.g. R@5, R², AUROC). */
  task_head_metrics?: Record<string, number>;
  /** The experiment/validation record id in benchmark_archive.json. */
  experiment_id?: string;
  /** When this provenance was generated. */
  timestamp: string;
  /** Full block weights for traceability. */
  block_weights: typeof JOINT_2312_BLOCK_WEIGHTS;
  /** Component dimensions for traceability. */
  component_dims: typeof JOINT_2312_COMPONENT_DIMS;
}

/**
 * Build a ServiceProvenance by reading the canonical artifact SHAs from
 * `joint2312Provenance()` at runtime. This guarantees the reported digests
 * exactly match the SHA-256 digests that were verified during embedding
 * computation (CBraMod `c128ccfd…`, V2 `18644de1…`, EEGPT `a92daf44…`).
 *
 * The PCA block is a deterministic band-power projection (no artifact file),
 * so its "SHA" is a provenance label derived from the feature specification.
 *
 * @param opts — Service/head-specific fields.
 * @returns A complete ServiceProvenance record.
 */
export function buildServiceProvenance(opts: {
  service: string;
  serviceVersion: string;
  taskHeadId: string;
  taskHeadVersion?: string;
  taskHeadSha256?: string;
  taskHeadDataset?: string;
  taskHeadMetrics?: Record<string, number>;
  experimentId?: string;
}): ServiceProvenance {
  // Read the canonical provenance from the Joint-2312 runtime. This calls
  // `joint2312Provenance()` which loads the manifest + verifies SHA-256
  // digests against the on-disk artifacts. The `pca` field is not in the
  // manifest (PCA is a pure-JS deterministic projection), so we derive a
  // stable label.
  const embeddingProv = joint2312Provenance();

  return {
    service: opts.service,
    service_version: opts.serviceVersion,
    embedding_model: JOINT_2312_MODEL_ID,
    embedding_dim: JOINT_2312_EMBEDDING_DIM,
    artifact_shas: {
      cbramod: embeddingProv.cbramod_sha256,
      v2: embeddingProv.v2_sha256,
      pca: derivePCASHA(),
      eegpt: embeddingProv.eegpt_sha256,
    },
    task_head_id: opts.taskHeadId,
    task_head_version: opts.taskHeadVersion,
    task_head_sha256: opts.taskHeadSha256,
    task_head_dataset: opts.taskHeadDataset,
    task_head_metrics: opts.taskHeadMetrics,
    experiment_id: opts.experimentId,
    timestamp: new Date().toISOString(),
    block_weights: JOINT_2312_BLOCK_WEIGHTS,
    component_dims: JOINT_2312_COMPONENT_DIMS,
  };
}

/**
 * PCA is a deterministic band-power projection (no artifact file), so we
 * derive a stable identifier from the feature definition: 5 bands × 22
 * channels = 110 input features → 32 PCA components.
 *
 * This is a provenance label, not a cryptographic hash — it identifies the
 * deterministic PCA transform version so that results are auditable.
 */
export function derivePCASHA(): string {
  const spec = "pca-bandpower-{bands:5,channels:22,features:110,dim:32}";
  let hash = 0;
  for (let i = 0; i < spec.length; i++) {
    const code = spec.charCodeAt(i);
    hash = (hash << 5) - hash + code;
    hash |= 0; // to 32-bit
  }
  return Math.abs(hash).toString(16).padStart(64, "0");
}
