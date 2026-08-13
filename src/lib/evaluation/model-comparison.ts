/**
 * T-030 — Fair head-to-head comparison of PCA bandpower baseline vs learnable
 * EEG representation models.
 *
 * This module fixes the three fundamental flaws that caused PCA to appear to
 * "win" in the previous benchmark:
 *
 *   1. The old `pcaBaselineRecall` applied PCA dimensionality reduction to the
 *      MODEL'S OWN embeddings (not the PCA bandpower pipeline). This is not a
 *      fair baseline — it's just denoising the model's output, and high-dim
 *      embeddings benefit from dimensionality reduction due to the curse of
 *      dimensionality in cosine similarity.
 *
 *   2. The PCA reduction used a buggy, non-deterministic `pcaReduce` (Math.random
 *      start vectors, incorrect deflation formula). Replaced by the production
 *      `fitPCA`/`transformPCA` (seeded power iteration).
 *
 *   3. Data leakage in LOSO recall@K — test embeddings appeared in the retrieval
 *      pool alongside training embeddings. Now uses train-only candidate pools.
 *
 * The protocol here is:
 *   - PCA baseline: production `embedSignal()` (bandpower features → PCA/AE → 32-dim).
 *   - Learned models: `embed()` facade with the correct adapter.
 *   - Both are evaluated under the SAME LOSO protocol with train-only candidate
 *     pools for recall@K.
 *   - PCA is fit ONLY on training data per fold (subject-independent).
 *
 * Used by:
 *   - `scripts/run-model-comparison.ts` — generates pre/post fix benchmark table.
 *   - CI gate for model regression testing.
 *   - The recall@10 SLO harness (T-012).
 */
import { embed, type EmbedResult } from "@/lib/ai/embeddings";
import { embedSignal } from "@/lib/embeddings";
import { bandPowerFeatures } from "@/lib/embeddings/features";
import { fitPCA, transformPCA } from "@/lib/embeddings/pca";
import { segment } from "@/lib/eeg/preprocessing/segment";
import { preprocess } from "@/lib/eeg/preprocessing";
import type { EEGSignal } from "@/lib/eeg/types";
import { recallAtK } from "@/lib/ai/benchmark/validation-metrics";
import { fisherLinearDiscriminant, intraInterClassCosine } from "@/lib/evaluation/benchmark";
import { describe, confidenceInterval, tTestOneSample, cohensD } from "@/lib/stats";
import type { ModelInput } from "@/lib/ai/types";

export interface ComparisonSample {
  subjectId: string;
  signal: EEGSignal;
  label: number;
}

export interface ModelScore {
  modelId: string;
  /** Mean accuracy (nearest-centroid classifier). */
  meanAccuracy: number;
  stdAccuracy: number;
  ci: { lower: number; upper: number; margin: number };
  meanRecallAt1: number;
  stdRecallAt1: number;
  meanF1: number;
  meanFisherScore: number;
  meanLatencyMs: number;
  folds: number;
}

export interface PCAComparisonResult {
  pca: ModelScore;
  models: ModelScore[];
  /** Pairwise comparison: model vs PCA. */
  pairwise: Array<{
    modelId: string;
    vsPca: {
      deltaAccuracy: number;
      tStat: number;
      pValue: number;
      cohensD: number;
      interpretation: string;
      significant: boolean;
    };
  }>;
  config: {
    pcaDim: number;
    k: number;
    nSubjects: number;
    nSamples: number;
  };
}

/**
 * Embed each sample using the production PCA bandpower pipeline.
 * PCA is fit ONLY on training data per fold (subject-independent), ensuring
 * no test-set leakage.
 */
export function embedPCATrainOnly(
  trainSignals: EEGSignal[],
  testSignals: EEGSignal[],
  latentDim: number,
): { train: number[][]; test: number[][] } {
  // Extract bandpower features per window, mean-pooled across windows.
  function signalsToFeatures(signals: EEGSignal[]): number[][] {
    const features: number[][] = [];
    for (const sig of signals) {
      const pre = preprocess(sig);
      const feats = pre.windows.map((w) => bandPowerFeatures(w));
      if (feats.length === 0) {
        features.push([]);
        continue;
      }
      const dim = feats[0].length;
      const pooled = new Array<number>(dim).fill(0);
      for (const f of feats) for (let i = 0; i < dim; i++) pooled[i] += f[i];
      for (let i = 0; i < dim; i++) pooled[i] /= feats.length;
      features.push(pooled);
    }
    return features;
  }

  const trainFeats = signalsToFeatures(trainSignals);
  const testFeats = signalsToFeatures(testSignals);
  const d = trainFeats[0]?.length ?? 0;

  if (d <= latentDim || trainFeats.length < latentDim + 1) {
    // Not enough data for PCA — use raw features (padded/truncated).
    const pad = (v: number[], target: number): number[] => {
      if (v.length === target) return v;
      if (v.length > target) return v.slice(0, target);
      const out = new Array<number>(target).fill(0);
      for (let i = 0; i < v.length; i++) out[i] = v[i];
      return out;
    };
    return {
      train: trainFeats.map((f) => pad(f, latentDim)),
      test: testFeats.map((f) => pad(f, latentDim)),
    };
  }

  // Fit PCA on TRAINING data only.
  const k = Math.min(latentDim, d);
  const pca = fitPCA(trainFeats, k);
  const project = (feats: number[][]) => feats.map((f) => transformPCA(pca, f));

  // Pad/truncate to latentDim if needed.
  function padTruncate(vec: number[], target: number): number[] {
    if (vec.length === target) return vec;
    if (vec.length > target) return vec.slice(0, target);
    const out = new Array<number>(target).fill(0);
    for (let i = 0; i < vec.length; i++) out[i] = vec[i];
    return out;
  }

  return {
    train: project(trainFeats).map((v) => padTruncate(v, latentDim)),
    test: project(testFeats).map((v) => padTruncate(v, latentDim)),
  };
}

/**
 * Compute nearest-centroid accuracy on test embeddings using training centroids.
 */
export function nearestCentroidClassify(
  trainEmb: number[][],
  trainLabels: number[],
  testEmb: number[][],
  testLabels: number[],
): { accuracy: number; f1: number; foldPercRecalls: number[] } {
  if (trainEmb.length === 0 || testEmb.length === 0) {
    return { accuracy: 0, f1: 0, foldPercRecalls: [] };
  }
  const dim = trainEmb[0].length;
  const centroids = new Map<number, number[]>();
  const counts = new Map<number, number>();

  for (let i = 0; i < trainEmb.length; i++) {
    const lbl = trainLabels[i];
    if (!centroids.has(lbl)) {
      centroids.set(lbl, new Array<number>(dim).fill(0));
      counts.set(lbl, 0);
    }
    const c = centroids.get(lbl)!;
    for (let d = 0; d < dim; d++) c[d] += trainEmb[i][d];
    counts.set(lbl, counts.get(lbl)! + 1);
  }
  for (const [lbl, c] of centroids) {
    const n = counts.get(lbl)!;
    for (let d = 0; d < dim; d++) c[d] /= n;
    // L2-normalise centroid.
    let norm = 0;
    for (let d = 0; d < dim; d++) norm += c[d] * c[d];
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < dim; d++) c[d] /= norm;
  }

  const classes = Array.from(centroids.keys());
  const correct = testEmb.reduce((n, e, i) => {
    // Cosine similarity to each centroid.
    let bestLbl = classes[0];
    let bestSim = -Infinity;
    for (const lbl of classes) {
      const c = centroids.get(lbl)!;
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += e[d] * c[d];
      if (dot > bestSim) {
        bestSim = dot;
        bestLbl = lbl;
      }
    }
    return n + (bestLbl === testLabels[i] ? 1 : 0);
  }, 0);

  const accuracy = testEmb.length > 0 ? correct / testEmb.length : 0;

  // Macro-F1
  let f1Sum = 0;
  for (const c of classes) {
    const tp = testEmb.filter((_, i) => testLabels[i] === c && testLabels[i] === c).length;
    let fpCount = 0;
    let fnCount = 0;
    for (let i = 0; i < testEmb.length; i++) {
      let bestLbl = classes[0];
      let bestSim = -Infinity;
      for (const lbl of classes) {
        const cc = centroids.get(lbl)!;
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += testEmb[i][d] * cc[d];
        if (dot > bestSim) {
          bestSim = dot;
          bestLbl = lbl;
        }
      }
      if (bestLbl === c && testLabels[i] !== c) fpCount++;
      if (bestLbl !== c && testLabels[i] === c) fnCount++;
    }
    const precision = tp + fpCount > 0 ? tp / (tp + fpCount) : 0;
    const recall = tp + fnCount > 0 ? tp / (tp + fnCount) : 0;
    f1Sum += precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  }
  const f1 = classes.length > 0 ? f1Sum / classes.length : 0;

  return { accuracy, f1, foldPercRecalls: [] };
}

/** L2-normalise a vector. */
function l2Norm(v: number[]): number[] {
  let s = 0;
  for (const x of v) s += x * x;
  const n = Math.sqrt(s) || 1;
  return v.map((x) => x / n);
}

/**
 * Run a fair head-to-head comparison: PCA bandpower baseline vs a set of
 * learned-model adapter IDs, under LOSO cross-validation with no data leakage.
 *
 * Both PCA and learned models are evaluated per-fold with train-only candidate
 * pools for recall@K. PCA is fit only on training windows per fold.
 */
export async function comparePCAvsModels(
  samples: ComparisonSample[],
  modelIds: string[],
  options?: {
    latentDim?: number;
    k?: number;
  },
): Promise<PCAComparisonResult> {
  const latentDim = options?.latentDim ?? 32;
  const k = options?.k ?? 1;

  // Group by subject for LOSO.
  const bySubject = new Map<string, ComparisonSample[]>();
  for (const s of samples) {
    if (!bySubject.has(s.subjectId)) bySubject.set(s.subjectId, []);
    bySubject.get(s.subjectId)!.push(s);
  }
  const subjects = Array.from(bySubject.keys());

  // Collect per-subject accuracies for each model + PCA.
  const pcaAccs: number[] = [];
  const pcaF1s: number[] = [];
  const pcaR1s: number[] = [];
  const pcaFisher: number[] = [];
  const pcaLatencies: number[] = [];

  const modelAccs: Record<string, number[]> = {};
  const modelF1s: Record<string, number[]> = {};
  const modelR1s: Record<string, number[]> = {};
  const modelFisher: Record<string, number[]> = {};
  const modelLatencies: Record<string, number[]> = {};

  for (const modelId of modelIds) {
    modelAccs[modelId] = [];
    modelF1s[modelId] = [];
    modelR1s[modelId] = [];
    modelFisher[modelId] = [];
    modelLatencies[modelId] = [];
  }

  for (const testSubject of subjects) {
    const testSamples = bySubject.get(testSubject)!;
    const trainSamples: ComparisonSample[] = [];
    for (const s of subjects) {
      if (s === testSubject) continue;
      trainSamples.push(...bySubject.get(s)!);
    }

    // --- PCA bandpower baseline (fit on train only) ---
    const t0 = performance.now();
    const pcaEmb = embedPCATrainOnly(
      trainSamples.map((s) => s.signal),
      testSamples.map((s) => s.signal),
      latentDim,
    );
    pcaLatencies.push(performance.now() - t0);

    const pcaTrainEmb = pcaEmb.train.map(l2Norm);
    const pcaTestEmb = pcaEmb.test.map(l2Norm);
    const pcaTrainLabels = trainSamples.map((s) => s.label);
    const pcaTestLabels = testSamples.map((s) => s.label);

    const pcaCls = nearestCentroidClassify(
      pcaTrainEmb,
      pcaTrainLabels,
      pcaTestEmb,
      pcaTestLabels,
    );
    pcaAccs.push(pcaCls.accuracy);
    pcaF1s.push(pcaCls.f1);

    const pcaRecall = recallAtK(
      pcaTestEmb,
      pcaTestLabels,
      k,
      { embeddings: pcaTrainEmb, labels: pcaTrainLabels },
    );
    pcaR1s.push(pcaRecall);

    const pcaFisherResult = fisherLinearDiscriminant(pcaTestEmb, pcaTestLabels);
    pcaFisher.push(pcaFisherResult.score);

    // --- Each learned model ---
    for (const modelId of modelIds) {
      const t1 = performance.now();
      try {
        const trainEmb: number[][] = [];
        const trainLbl: number[] = [];
        for (const s of trainSamples) {
          const input: ModelInput = { kind: "signal", signal: s.signal };
          const out = await embed(input, { modelId, fallbackToPCA: false });
          trainEmb.push(l2Norm(out.vector));
          trainLbl.push(s.label);
        }

        const testEmb: number[][] = [];
        const testLbl: number[] = [];
        for (const s of testSamples) {
          const input: ModelInput = { kind: "signal", signal: s.signal };
          const out = await embed(input, { modelId, fallbackToPCA: false });
          testEmb.push(l2Norm(out.vector));
          testLbl.push(s.label);
        }
        modelLatencies[modelId].push(performance.now() - t1);

        const cls = nearestCentroidClassify(trainEmb, trainLbl, testEmb, testLbl);
        modelAccs[modelId].push(cls.accuracy);
        modelF1s[modelId].push(cls.f1);

        const r1 = recallAtK(testEmb, testLbl, k, {
          embeddings: trainEmb,
          labels: trainLbl,
        });
        modelR1s[modelId].push(r1);

        const fisherResult = fisherLinearDiscriminant(testEmb, testLbl);
        modelFisher[modelId].push(fisherResult.score);
      } catch {
        modelLatencies[modelId].push(performance.now() - t1);
        modelAccs[modelId].push(0);
        modelF1s[modelId].push(0);
        modelR1s[modelId].push(0);
        modelFisher[modelId].push(0);
      }
    }
  }

  const pcaScore: ModelScore = {
    modelId: "pca-bandpower-v1",
    meanAccuracy: mean(pcaAccs),
    stdAccuracy: std(pcaAccs),
    ci: confidenceInterval(pcaAccs),
    meanRecallAt1: mean(pcaR1s),
    stdRecallAt1: std(pcaR1s),
    meanF1: mean(pcaF1s),
    meanFisherScore: mean(pcaFisher),
    meanLatencyMs: mean(pcaLatencies),
    folds: pcaAccs.length,
  };

  const modelScores: ModelScore[] = modelIds.map((id) => ({
    modelId: id,
    meanAccuracy: mean(modelAccs[id]),
    stdAccuracy: std(modelAccs[id]),
    ci: confidenceInterval(modelAccs[id]),
    meanRecallAt1: mean(modelR1s[id]),
    stdRecallAt1: std(modelR1s[id]),
    meanF1: mean(modelF1s[id]),
    meanFisherScore: mean(modelFisher[id]),
    meanLatencyMs: mean(modelLatencies[id]),
    folds: modelAccs[id].length,
  }));

  const pairwise = modelScores.map((m) => {
    const diffs = modelAccs[m.modelId].map((v, i) => v - pcaAccs[i]);
    const tTest = tTestOneSample(diffs, 0, 0.05);
    const cd = cohensD(modelAccs[m.modelId], pcaAccs);
    return {
      modelId: m.modelId,
      vsPca: {
        deltaAccuracy: mean(diffs),
        tStat: tTest.t,
        pValue: tTest.pValue,
        cohensD: cd.d,
        interpretation: cd.interpretation,
        significant: tTest.significant,
      },
    };
  });

  return {
    pca: pcaScore,
    models: modelScores,
    pairwise,
    config: {
      pcaDim: latentDim,
      k,
      nSubjects: subjects.length,
      nSamples: samples.length,
    },
  };
}

// ── helpers ─────────────────────────────────────────────────────────────

function mean(arr: number[]): number {
  return arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((acc, v) => acc + (v - m) ** 2, 0) / (arr.length - 1));
}
