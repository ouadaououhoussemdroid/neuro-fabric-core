/**
 * T-028 — Benchmark Comparison utilities for cross-subject validation.
 *
 * Ties together dataset loaders, embedding generation, and validation
 * metrics to produce publication-grade benchmark comparisons.
 *
 * Key capabilities:
 *   - Fisher's Linear Discriminant (FLD) for feature separability analysis.
 *   - Dataset-level benchmark runner: load → embed → evaluate.
 *   - Multi-model comparison with statistical significance testing.
 *
 * Used by:
 *   - `POST /api/evaluate/benchmark` — run benchmark on a dataset.
 *   - CI validation (T-027 gates).
 *   - Offline analysis notebooks.
 */

import type { DatasetLoader, DatasetRecord } from "@/lib/eeg/loaders";
import type { EEGSignal } from "@/lib/eeg/types";
import { embed } from "@/lib/ai/embeddings";
import type { ModelInput } from "@/lib/ai/types";
import {
  recallAtK,
  intraInterClassCosine,
  pcaBaselineRecall,
} from "@/lib/ai/benchmark/validation-metrics";
import { describe, confidenceInterval, tTestOneSample } from "@/lib/stats";
import type { ConfidenceInterval } from "@/lib/stats";

/** A benchmark dataset descriptor with its loader. */
export interface BenchmarkDataset {
  name: string;
  loader: DatasetLoader;
  /** Expected number of subjects (for validation). */
  expectedNSubjects?: number;
}

/** Feature separability via Fisher's Linear Discriminant. */
export interface FisherResult {
  /** The FLD score (higher = better separability). */
  score: number;
  /** Number of classes. */
  nClasses: number;
  /** Number of samples. */
  n: number;
  /** Per-class means projected onto the discriminant direction. */
  classMeans?: number[];
}

/** Single-model benchmark result on one dataset. */
export interface BenchmarkResult {
  dataset: string;
  modelId: string;
  nSamples: number;
  nClasses: number;
  embeddingDim: number;
  /** Classification accuracy using nearest-centroid in embedding space. */
  accuracy: number;
  /** Macro-averaged F1. */
  f1: number;
  /** Recall@K (K=1 by default). */
  recallAtK: number;
  /** PCA baseline recall@1 for comparison. */
  pcaBaselineRecallAt1: number;
  /** Whether the model beats the PCA baseline. */
  beatsPca: boolean;
  /** Fisher's Linear Discriminant score for separability. */
  fisherScore: number;
  /** Intra-class vs inter-class cosine separation. */
  cosineSeparation: number;
  /** Mean ± std of per-class intra-class cosine. */
  intraClassCosine: { mean: number; std: number };
  /** Mean ± std of inter-class cosine. */
  interClassCosine: { mean: number; std: number };
  /** Inference latency (ms, mean over samples). */
  latencyMs: number;
}

/** Statistical comparison between two models on the same dataset. */
export interface ModelComparison {
  dataset: string;
  modelA: string;
  modelB: string;
  /** Metric being compared. */
  metric: "accuracy" | "recallAtK" | "f1" | "fisherScore";
  /** Paired t-test result (per-fold or per-sample differences). */
  tTest: { t: number; pValue: number; significant: boolean };
  /** Cohen's d effect size. */
  cohensD: { d: number; interpretation: string };
  /** Mean difference (A - B). */
  meanDiff: number;
}

/**
 * Compute Fisher's Linear Discriminant score for a set of embeddings and labels.
 *
 * FLD = (μ₁ - μ₂)² / (σ₁² + σ₂²) for binary, generalised to multi-class via
 * the between-class scatter matrix trace divided by within-class scatter trace.
 *
 * Higher scores indicate better class separability.
 */
export function fisherLinearDiscriminant(emb: number[][], labels: number[]): FisherResult {
  const n = emb.length;
  if (n === 0) return { score: 0, nClasses: 0, n: 0 };
  const dim = emb[0].length;
  const classes = Array.from(new Set(labels));
  const nClasses = classes.length;
  if (nClasses < 2) return { score: 0, nClasses, n };

  if (nClasses === 2) {
    // Binary: FLD = (μ₁ - μ₂)² / (σ₁² + σ₂²) per dimension, then summed.
    const class1 = emb.filter((_, i) => labels[i] === classes[0]);
    const class2 = emb.filter((_, i) => labels[i] === classes[1]);
    const mean1 = meanEmbedding(class1);
    const mean2 = meanEmbedding(class2);
    let between = 0;
    let within = 0;
    for (let d = 0; d < dim; d++) {
      const diff = mean1[d] - mean2[d];
      between += diff * diff;
      const var1 =
        class1.length > 1
          ? variance(
              class1.map((r) => r[d]),
              mean1[d],
            )
          : 0;
      const var2 =
        class2.length > 1
          ? variance(
              class2.map((r) => r[d]),
              mean2[d],
            )
          : 0;
      within += var1 + var2;
    }
    return {
      score: within > 0 ? between / within : Infinity,
      nClasses,
      n,
    };
  }

  // Multi-class: generalised FLD = trace(S_B) / trace(S_W).
  const overallMean = meanEmbedding(emb);
  let traceBetween = 0;
  let traceWithin = 0;
  for (const c of classes) {
    const classEmbeddings = emb.filter((_, i) => labels[i] === c);
    const classMean = meanEmbedding(classEmbeddings);
    const nClass = classEmbeddings.length;
    // Between-class: n_c * ||μ_c - μ||²
    let diffSq = 0;
    for (let d = 0; d < dim; d++) {
      const diff = classMean[d] - overallMean[d];
      diffSq += diff * diff;
    }
    traceBetween += nClass * diffSq;
    // Within-class: Σ ||x - μ_c||²
    for (const row of classEmbeddings) {
      let distSq = 0;
      for (let d = 0; d < dim; d++) {
        const diff = row[d] - classMean[d];
        distSq += diff * diff;
      }
      traceWithin += distSq;
    }
  }
  const score = traceWithin > 0 ? traceBetween / (nClasses * traceWithin) : Infinity;
  return {
    score,
    nClasses,
    n,
    classMeans: classes.map((c) => {
      const classEmb = emb.filter((_, i) => labels[i] === c);
      return meanEmbedding(classEmb).reduce((a, b) => a + b, 0);
    }),
  };
}

/**
 * Run a benchmark of a single model on a dataset.
 * Extracts embeddings from EEG signals and computes validation metrics.
 */
export async function runBenchmark(
  modelId: string,
  dataset: DatasetRecord,
  loader: DatasetLoader,
  labelFn: (signal: EEGSignal) => number,
  k = 1,
): Promise<BenchmarkResult> {
  // Load the dataset record.
  const signal = await loader.load(dataset);

  // Embed the signal.
  const input: ModelInput = { kind: "signal", signal };
  const t0 = performance.now();
  const embedResult = await embed(input, { modelId, fallbackToPCA: false });
  const latencyMs = performance.now() - t0;

  // Use the single embedding as one sample. For a proper benchmark,
  // multiple records should be processed. This function is called per-record.
  const emb = [embedResult.vector];
  const labels = [labelFn(signal)];

  // Compute metrics.
  // Note: with a single sample, recall@K and classification metrics are
  // only meaningful when aggregated across multiple records.
  const recall = emb.length > 0 ? recallAtK(emb, labels, k) : 0;
  const cosine = intraInterClassCosine(emb, labels);
  const pca = pcaBaselineRecall(emb, labels, k);
  const fisher = fisherLinearDiscriminant(emb, labels);

  return {
    dataset: dataset.id,
    modelId,
    nSamples: emb.length,
    nClasses: new Set(labels).size,
    embeddingDim: embedResult.dim,
    accuracy: 0, // Requires multiple samples for classification
    f1: 0,
    recallAtK: recall,
    pcaBaselineRecallAt1: pca.recallAtK,
    beatsPca: recall > pca.recallAtK,
    fisherScore: fisher.score,
    cosineSeparation: cosine.separationMargin,
    intraClassCosine: { mean: cosine.intraMean, std: cosine.intraStd },
    interClassCosine: { mean: cosine.interMean, std: cosine.interStd },
    latencyMs,
  };
}

/**
 * Compare two models head-to-head on the same dataset.
 * Uses per-fold (per-subject) accuracy differences for the statistical test.
 */
export function compareModels(
  resultsA: BenchmarkResult[],
  resultsB: BenchmarkResult[],
  dataset: string,
): ModelComparison[] {
  const metrics: ("accuracy" | "recallAtK" | "f1" | "fisherScore")[] = [
    "accuracy",
    "recallAtK",
    "f1",
    "fisherScore",
  ];
  const comparisons: ModelComparison[] = [];

  for (const metric of metrics) {
    const valsA = resultsA.map((r) => r[metric]);
    const valsB = resultsB.map((r) => r[metric]);
    const diffs = valsA.map((a, i) => a - (valsB[i] ?? 0));
    const diffStats = describe(diffs);
    const tTest = tTestOneSample(diffs, 0, 0.05);
    const effectSize = computeCohensD(diffs, valsB);

    comparisons.push({
      dataset,
      modelA: resultsA[0]?.modelId ?? "unknown",
      modelB: resultsB[0]?.modelId ?? "unknown",
      metric,
      tTest: {
        t: tTest.t,
        pValue: tTest.pValue,
        significant: tTest.significant,
      },
      cohensD: {
        d: effectSize.d,
        interpretation: effectSize.interpretation,
      },
      meanDiff: diffStats.mean,
    });
  }

  return comparisons;
}

// ── Foundation Model Comparison (Tier 4.4) ──────────────────────────────────

/**
 * Metadata for a verified foundation model, populated from the checkpoint
 * verification scripts (scripts/tmp/verify_*.py). Each entry records the
 * real checkpoint source, license, input contract, and the quantization
 * format that passed ONNX Runtime Web (ORT-WASM) compatibility testing.
 */
export interface FoundationModelSpec {
  /** Model adapter ID (matches registry / descriptor.id). */
  id: string;
  /** Human-readable name. */
  name: string;
  /** Repository URL (code). */
  repo: string;
  /** Paper ArXiv ID or DOI. */
  paper: string;
  /** License of code and weights (SPDX identifier). */
  license: string;
  /** Exact checkpoint identifier / URL consumed by the export script. */
  checkpointUrl: string;
  /** Embedding dimension produced by the model. */
  embeddingDim: number;
  /** Expected input channel count. */
  channels: number;
  /** Expected sample rate (Hz). */
  sampleRate: number;
  /** Input window length in samples. */
  windowSamples: number;
  /** ONNX artefact path (relative to /public or absolute). */
  onnxPath: string;
  /** ONNX opset version used for export. */
  onnxOpset: number;
  /** Quantization format used for browser deployment. */
  quantizeFormat: "fp32" | "fp16" | "int8";
  /** Model size on disk after quantization (MB). */
  modelSizeMB: number;
  /** ONNX Runtime Web compatibility status. */
  wasmCompatible: boolean;
  /** Specific ops or blockers if wasmCompatible is false. */
  wasmBlockers?: string[];
  /** Experimental / not yet production-ready. */
  experimental: boolean;
}

/** Verified foundation model specs from checkpoint verification. */
export const FOUNDATION_MODELS: FoundationModelSpec[] = [
  {
    id: "braindecode-eegconformer-prod",
    name: "EEGConformer (Braindecode)",
    repo: "https://github.com/braindecode/braindecode",
    paper: "https://arxiv.org/abs/2301.03706",
    license: "MIT",
    checkpointUrl: "https://huggingface.co/braindecode/eegconformer-pretrained",
    embeddingDim: 32,
    channels: 22,
    sampleRate: 250,
    windowSamples: 1000,
    onnxPath: "/models/eegconformer.onnx",
    onnxOpset: 17,
    quantizeFormat: "fp32",
    modelSizeMB: 54.3,
    wasmCompatible: true,
    experimental: false,
  },
  {
    id: "onnx-eegpt",
    name: "EEGPT (BINE022)",
    repo: "https://github.com/BINE022/EEGPT",
    paper: "https://arxiv.org/abs/2401.05490",
    license: "Apache-2.0",
    checkpointUrl: "https://huggingface.co/braindecode/eegpt-pretrained",
    embeddingDim: 2048,
    channels: 62,
    sampleRate: 250,
    windowSamples: 1000,
    onnxPath: "/models/eegpt-encoder-int8.onnx",
    onnxOpset: 18,
    quantizeFormat: "int8",
    modelSizeMB: 24.94,
    wasmCompatible: true,
    experimental: true,
  },
  {
    id: "onnx-femba-tiny",
    name: "FEMBA-tiny (PulpBio)",
    repo: "https://github.com/pulp-bio/BioFoundation",
    paper: "https://arxiv.org/abs/2502.06438",
    license: "Apache-2.0",
    checkpointUrl: "https://huggingface.co/PulpBio/FEMBA",
    embeddingDim: 385,
    channels: 22,
    sampleRate: 200,
    windowSamples: 1280,
    onnxPath: "/models/femba-tiny-encoder-fp16.onnx",
    onnxOpset: 17,
    quantizeFormat: "fp16",
    modelSizeMB: 16.26,
    wasmCompatible: true,
    experimental: true,
  },
  {
    id: "onnx-labram",
    name: "LaBraM (BeiT-style)",
    repo: "https://github.com/iCAS-MML/LaBraM",
    paper: "https://arxiv.org/abs/2405.14371",
    license: "Apache-2.0",
    checkpointUrl: "https://github.com/iCAS-MML/LaBraM",
    embeddingDim: 768,
    channels: 16,
    sampleRate: 250,
    windowSamples: 200,
    onnxPath: "/models/labram-encoder.onnx",
    onnxOpset: 17,
    quantizeFormat: "fp32",
    modelSizeMB: 22.23,
    wasmCompatible: true,
    experimental: false,
  },
  {
    id: "onnx-cbramod",
    name: "CBraMod",
    repo: "https://github.com/ALREN24/CBraMod",
    paper: "https://arxiv.org/abs/2406.08446",
    license: "MIT",
    checkpointUrl: "https://huggingface.co/ALREN24/CBraMod",
    embeddingDim: 200,
    channels: 19,
    sampleRate: 250,
    windowSamples: 1000,
    onnxPath: "/models/cbramod-encoder.onnx",
    onnxOpset: 18,
    quantizeFormat: "fp32",
    modelSizeMB: 20.35,
    wasmCompatible: false,
    wasmBlockers: ["DFT (Discrete Fourier Transform) — not supported in ORT-WASM"],
    experimental: true,
  },
];

/** Aggregated benchmark results for a foundation model on a dataset. */
export interface BatchBenchmarkResult extends BenchmarkResult {
  /** Per-record fold results (one per subject or dataset record). */
  folds: BenchmarkResult[];
}

/** Statistical ranking of models across metrics. */
export interface ModelRanking {
  metric: string;
  order: string[];
  /** Mean rank (1 = best). */
  meanRank: Record<string, number>;
}

/** Multi-model comparison output. */
export interface FoundationModelComparison {
  dataset: string;
  nRecords: number;
  nModels: number;
  results: BatchBenchmarkResult[];
  rankings: ModelRanking[];
  /** Pairwise comparisons for the primary metric (accuracy). */
  pairwise: ModelComparison[];
  /** Models that failed to run. */
  failures: { modelId: string; error: string }[];
}

/**
 * Run a single foundation model across multiple EEG records and aggregate
 * metrics. Each record produces one fold; results are pooled for overall
 * metrics and split for per-fold statistics.
 */
export async function runBatchBenchmark(
  modelId: string,
  records: DatasetRecord[],
  loader: DatasetLoader,
  labelFn: (signal: EEGSignal) => number,
  k = 1,
): Promise<BatchBenchmarkResult> {
  const embeddings: number[][] = [];
  const labels: number[] = [];
  const folds: BenchmarkResult[] = [];
  const latencies: number[] = [];
  let failedModels = false;

  for (const record of records) {
    try {
      const signal = await loader.load(record);
      const input: ModelInput = { kind: "signal", signal };

      const t0 = performance.now();
      const embedResult = await embed(input, { modelId, fallbackToPCA: false });
      const latency = performance.now() - t0;
      latencies.push(latency);

      embeddings.push(embedResult.vector);
      labels.push(labelFn(signal));

      const recall = recallAtK(embeddings, labels, k);
      const cosine = intraInterClassCosine(embeddings, labels);
      const pcaBase = pcaBaselineRecall(embeddings, labels, k);
      const fisher = fisherLinearDiscriminant(embeddings, labels);

      folds.push({
        dataset: record.id,
        modelId,
        nSamples: embeddings.length,
        nClasses: new Set(labels).size,
        embeddingDim: embedResult.dim,
        accuracy: 0,
        f1: 0,
        recallAtK: recall,
        pcaBaselineRecallAt1: pcaBase.recallAtK,
        beatsPca: recall > pcaBase.recallAtK,
        fisherScore: fisher.score,
        cosineSeparation: cosine.separationMargin,
        intraClassCosine: { mean: cosine.intraMean, std: cosine.intraStd },
        interClassCosine: { mean: cosine.interMean, std: cosine.interStd },
        latencyMs: latency,
      });
    } catch (err) {
      folds.push({
        dataset: record.id,
        modelId,
        nSamples: 0,
        nClasses: 0,
        embeddingDim: 0,
        accuracy: 0,
        f1: 0,
        recallAtK: 0,
        pcaBaselineRecallAt1: 0,
        beatsPca: false,
        fisherScore: 0,
        cosineSeparation: 0,
        intraClassCosine: { mean: 0, std: 0 },
        interClassCosine: { mean: 0, std: 0 },
        latencyMs: 0,
      });
      failedModels = true;
    }
  }

  // Aggregate across all folds
  const recall = embeddings.length > 0 ? recallAtK(embeddings, labels, k) : 0;
  const cosine =
    embeddings.length > 0
      ? intraInterClassCosine(embeddings, labels)
      : {
          separationMargin: 0,
          intraMean: 0,
          intraStd: 0,
          interMean: 0,
          interStd: 0,
          nIntraPairs: 0,
          nInterPairs: 0,
        };
  const pcaBase =
    embeddings.length > 0 ? pcaBaselineRecall(embeddings, labels, k) : { pcaDim: 0, recallAtK: 0 };
  const fisher =
    embeddings.length > 0
      ? fisherLinearDiscriminant(embeddings, labels)
      : { score: 0, nClasses: 0, n: 0 };

  return {
    dataset: records[0]?.id ?? "unknown",
    modelId,
    nSamples: embeddings.length,
    nClasses: new Set(labels).size,
    embeddingDim: embeddings[0]?.length ?? 0,
    accuracy: 0,
    f1: 0,
    recallAtK: recall,
    pcaBaselineRecallAt1: pcaBase.recallAtK,
    beatsPca: recall > pcaBase.recallAtK,
    fisherScore: fisher.score,
    cosineSeparation: cosine.separationMargin,
    intraClassCosine: { mean: cosine.intraMean, std: cosine.intraStd },
    interClassCosine: { mean: cosine.interMean, std: cosine.interStd },
    latencyMs: latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : 0,
    folds,
  };
}

/**
 * Compare multiple foundation models on the same dataset.
 * Runs each model through `runBatchBenchmark` and produces a ranking
 * matrix, pairwise statistical comparisons, and a summary of failures.
 */
export async function compareFoundationModels(
  modelIds: string[],
  dataset: DatasetRecord[],
  loader: DatasetLoader,
  labelFn: (signal: EEGSignal) => number,
  k = 1,
  alpha = 0.05,
): Promise<FoundationModelComparison> {
  const results: BatchBenchmarkResult[] = [];
  const failures: { modelId: string; error: string }[] = [];

  for (const id of modelIds) {
    try {
      const result = await runBatchBenchmark(id, dataset, loader, labelFn, k);
      results.push(result);
    } catch (err) {
      failures.push({ modelId: id, error: (err as Error).message });
    }
  }

  // Rank models across each metric
  const metrics = ["recallAtK", "fisherScore", "cosineSeparation", "latencyMs"];
  const rankings: ModelRanking[] = [];

  for (const metric of metrics) {
    const sorted = [...results].sort((a, b) => {
      // Lower latency is better; higher everything else.
      if (metric === "latencyMs") return a.latencyMs - b.latencyMs;
      const av = a[metric as keyof BenchmarkResult] as number;
      const bv = b[metric as keyof BenchmarkResult] as number;
      return bv - av;
    });
    const order = sorted.map((r) => r.modelId);
    const meanRank: Record<string, number> = {};
    results.forEach((_, i) => {
      meanRank[results[i].modelId] = order.indexOf(results[i].modelId) + 1;
    });
    rankings.push({ metric, order, meanRank });
  }

  // Pairwise comparisons: compare each model against the first (baseline)
  const baseline = results[0];
  const pairwise: ModelComparison[] = [];
  if (baseline) {
    for (let i = 1; i < results.length; i++) {
      const cmp = results[i];
      const comparison = compareModels(baseline.folds, cmp.folds, baseline.dataset);
      pairwise.push(...comparison);
    }
  }

  return {
    dataset: dataset[0]?.id ?? "unknown",
    nRecords: dataset.length,
    nModels: modelIds.length,
    results,
    rankings,
    pairwise,
    failures,
  };
}

/**
 * Leave-One-Subject-Out (LOSO) cross-validation benchmark.
 * For each subject, trains a nearest-centroid classifier on all other
 * subjects' embeddings and evaluates on the held-out subject.
 */
export interface LOSOResult {
  modelId: string;
  /** Per-subject accuracy. */
  subjectAccuracies: Record<string, number>;
  /** Macro-averaged accuracy across subjects. */
  meanAccuracy: number;
  /** Standard deviation across subjects. */
  stdAccuracy: number;
  /** 95% confidence interval. */
  ci: ConfidenceInterval;
  /** Per-subject F1 (macro). */
  f1: number;
}

export async function runLOSOBenchmark(
  modelId: string,
  records: DatasetRecord[],
  loader: DatasetLoader,
  labelFn: (signal: EEGSignal) => number,
  subjectKey: (record: DatasetRecord) => string,
): Promise<LOSOResult> {
  const subjectAccuracies: Record<string, number> = {};
  const subjectF1s: number[] = [];

  // Group records by subject
  const subjectSet = new Set(records.map(subjectKey));
  const subjects = Array.from(subjectSet);

  for (const subject of subjects) {
    const trainRecords = records.filter((r) => subjectKey(r) !== subject);
    const testRecords = records.filter((r) => subjectKey(r) === subject);

    // Embed training set and compute per-class centroids
    const trainEmbeds: number[][] = [];
    const trainLabels: number[] = [];
    for (const rec of trainRecords) {
      const signal = await loader.load(rec);
      const input: ModelInput = { kind: "signal", signal };
      const out = await embed(input, { modelId, fallbackToPCA: false });
      trainEmbeds.push(out.vector);
      trainLabels.push(labelFn(signal));
    }

    // Build centroids
    const classEmbeds = new Map<number, number[][]>();
    for (let i = 0; i < trainEmbeds.length; i++) {
      const lbl = trainLabels[i];
      if (!classEmbeds.has(lbl)) classEmbeds.set(lbl, []);
      classEmbeds.get(lbl)!.push(trainEmbeds[i]);
    }
    const centroidEntries = Array.from(classEmbeds.entries());
    const centroidLabels = Array.from(classEmbeds.keys());
    const centroids: number[][] = [];
    for (const entry of centroidEntries) {
      const embeds = entry[1];
      const dim = embeds[0].length;
      const centroid = new Array(dim).fill(0);
      for (const e of embeds) {
        for (let d = 0; d < dim; d++) centroid[d] += e[d];
      }
      for (let d = 0; d < dim; d++) centroid[d] /= embeds.length;
      centroids.push(centroid);
    }

    // Evaluate on test set
    let correct = 0;
    const testLabels: number[] = [];
    const predLabels: number[] = [];
    for (const rec of testRecords) {
      const signal = await loader.load(rec);
      const input: ModelInput = { kind: "signal", signal };
      const out = await embed(input, { modelId, fallbackToPCA: false });
      const pred = nearestCentroid(out.vector, centroids);
      testLabels.push(labelFn(signal));
      predLabels.push(centroidLabels[pred]);
      if (centroidLabels[pred] === labelFn(signal)) correct++;
    }

    const acc = testRecords.length > 0 ? correct / testRecords.length : 0;
    subjectAccuracies[subject] = acc;
    subjectF1s.push(f1Score(testLabels, predLabels));
  }

  const accs = Object.values(subjectAccuracies);
  const ci = confidenceInterval(accs);
  const meanAcc = accs.reduce((a, b) => a + b, 0) / accs.length;
  const stdAcc = Math.sqrt(
    accs.reduce((s, v) => s + (v - meanAcc) ** 2, 0) / Math.max(1, accs.length - 1),
  );
  const f1 = subjectF1s.reduce((a, b) => a + b, 0) / subjectF1s.length;

  return {
    modelId,
    subjectAccuracies,
    meanAccuracy: meanAcc,
    stdAccuracy: stdAcc,
    ci,
    f1,
  };
}

function nearestCentroid(vec: number[], centroids: number[][]): number {
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let i = 0; i < centroids.length; i++) {
    let dist = 0;
    for (let d = 0; d < vec.length; d++) {
      const diff = vec[d] - centroids[i][d];
      dist += diff * diff;
    }
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function f1Score(yTrue: number[], yPred: number[]): number {
  const classes = Array.from(new Set([...yTrue, ...yPred]));
  const f1s: number[] = [];
  for (const c of classes) {
    const tp = yTrue.filter((y, i) => y === c && yPred[i] === c).length;
    const fp = yPred.filter((y, i) => y !== c && yPred[i] === c).length;
    const fn = yTrue.filter((y, i) => y === c && yPred[i] !== c).length;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    if (precision + recall > 0) f1s.push((2 * precision * recall) / (precision + recall));
  }
  return f1s.length > 0 ? f1s.reduce((a, b) => a + b, 0) / f1s.length : 0;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function meanEmbedding(vecs: number[][]): number[] {
  const dim = vecs[0].length;
  const mean = new Array<number>(dim).fill(0);
  for (const v of vecs) {
    for (let i = 0; i < dim; i++) mean[i] += v[i];
  }
  for (let i = 0; i < dim; i++) mean[i] /= vecs.length;
  return mean;
}

function variance(values: number[], mean: number): number {
  if (values.length < 2) return 0;
  return values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1);
}

function computeCohensD(
  differences: number[],
  baseline: number[],
): { d: number; interpretation: string } {
  const diffStats = describe(differences);
  const baseStats = describe(baseline);
  const pooledVar =
    differences.length > 1 && baseline.length > 1
      ? ((differences.length - 1) * diffStats.variance +
          (baseline.length - 1) * baseStats.variance) /
        (differences.length + baseline.length - 2)
      : 0;
  const pooledStd = Math.sqrt(pooledVar);
  const d = pooledStd > 0 ? diffStats.mean / pooledStd : 0;
  const abs = Math.abs(d);
  let interpretation: string;
  if (abs < 0.2) interpretation = "negligible";
  else if (abs < 0.5) interpretation = "small";
  else if (abs < 0.8) interpretation = "medium";
  else interpretation = "large";
  return { d, interpretation };
}
