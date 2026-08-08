/** T-028 — Evaluation utilities: LOSO, ground truth, and benchmark comparison. */
export { evaluateLOSO } from "./loso";
export type {
  LOFOSSample,
  LOSOFoldResult,
  LOSOAggregate,
  LOSOEvaluationResult,
  LOSOOptions,
} from "./loso";

export {
  correlatePredictions,
  groupLabels,
  summarizeAnnotations,
  type GroundTruthLabel,
  type GroundTruthSet,
  type AnnotationSummary,
  type CorrelationResult,
  type LabelType,
} from "./ground-truth";

export {
  fisherLinearDiscriminant,
  runBenchmark,
  compareModels,
  type BenchmarkDataset,
  type BenchmarkResult,
  type ModelComparison,
  type FisherResult,
} from "./benchmark";
