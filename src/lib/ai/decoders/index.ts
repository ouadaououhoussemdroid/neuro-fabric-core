/**
 * M31 Shared Service Layer — TaskHeadRegistry barrel export.
 *
 * Re-exports the generic registry + all service-specific head registrations.
 * Services register their heads at startup via their respective `registerXHeads()`
 * functions. The generic registry helpers (`registerTaskHead`, `getTaskHead`,
 * `hasTaskHead`, `listTaskHeads`, `getTaskHeadsByService`, `getDefaultTaskHead`)
 * are always available.
 *
 * Browser-safe: no `.server.ts` suffix; no onnxruntime import. Head descriptors
 * (metadata only) are safe to import in both the browser and the server.
 */
export {
  registerTaskHead,
  getTaskHead,
  hasTaskHead,
  listTaskHeads,
  getTaskHeadsByService,
  getDefaultTaskHead,
  serviceIdentity,
  type TaskHeadDescriptor,
  type InferenceTarget,
  type SHA256,
} from "./registry";

// Service-specific registries (browser-safe metadata only).
export { COGNITIVE_HEADS } from "./cognitive.registry";
export { ANOMALY_HEADS } from "./anomaly.registry";

// Re-export individual head descriptors for direct reference.
export {
  COGNITIVE_LINEAR_PROBE_JOINT_2312,
  COGNITIVE_LINEAR_PROBE_V2_32,
  COGNITIVE_MLP_PROBE_JOINT_2312,
  registerCognitiveHeads,
  getDefaultCognitiveHead,
} from "./cognitive.registry";
export {
  ANOMALY_MAHALANOBIS_PROBE_JOINT_2312,
  ANOMALY_MAHALANOBIS_PROBE_V2_32,
  registerAnomalyHeads,
  getDefaultAnomalyHead,
} from "./anomaly.registry";

// Browser-compatible cognitive decoder (V2-32 → workload).
export {
  browserCognitiveDecode,
  decodeFromV2Embedding,
  setBrowserProbeWeights,
  getBrowserProbeWeights,
  type BrowserCognitiveResult,
} from "./cognitive.browser";

// Browser-compatible anomaly detector (V2-32 → anomaly score).
export {
  browserAnomalyDetect,
  detectFromV2Embedding as detectAnomalyFromV2Embedding,
  setBrowserAnomalyWeights,
  getBrowserAnomalyWeights,
  type BrowserAnomalyResult,
} from "./anomaly.browser";

// M39 — Sleep staging + M40 — Sleep quality task heads + browser fallback.
export {
  SLEEP_STAGING_PROBE_JOINT_2312,
  SLEEP_STAGING_PROBE_V2_32,
  SLEEP_QUALITY_PROBE_JOINT_2312,
  SLEEP_QUALITY_PROBE_V2_32,
  SLEEP_HEADS,
  registerSleepHeads,
  getDefaultSleepHead,
  getDefaultSleepQualityHead,
} from "./sleep.registry";
export {
  browserSleepStage,
  detectSleepFromV2Embedding,
  setBrowserSleepWeights,
  getBrowserSleepWeights,
  browserSleepQuality,
  BROWSER_SLEEP_INPUT_DIM,
  BROWSER_SLEEP_OUTPUT_DIM,
  BROWSER_SLEEP_STAGES,
  BROWSER_SLEEP_CI_MARGIN,
  type BrowserSleepResult,
  type BrowserSleepQualityResult,
} from "./sleep.browser";
