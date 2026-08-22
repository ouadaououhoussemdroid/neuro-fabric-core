/**
 * M38 — Dataset loader barrel exports.
 *
 * Re-exports all dataset loaders and manifest utilities so callers can
 * import from a single path: `import { sleepEDFDataset, preprocessSleepEDF } from "@/lib/datasets"`.
 */
export { KNOWN_DATASETS } from "./manifest";
export {
  sleepEDFDataset,
  preprocessSleepEDF,
  expandSleepToEEGPT,
  parseSleepAnnotations,
  JOINT_2312_SAMPLE_RATE_HZ,
  JOINT_WINDOW_SEC,
  SLEEP_EPOCH_SEC,
  SLEEP_STAGES,
  SLEEP_STAGE_LABELS,
  SLEEP_STAGE_ID_TO_LABEL,
  SLEEP_STAGE_LABEL_TO_ID,
} from "./sleep-edf";
export type {
  SleepStage,
  SleepEpoch,
  SleepEEGPreprocessResult,
} from "./sleep-edf";
