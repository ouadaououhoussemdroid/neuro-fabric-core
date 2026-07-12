/**
 * T-010 — EEGConformer empirical validation metrics.
 *
 * Computes the same embedding-quality metrics as
 * `training/scripts/evaluate.py` (intra/inter-class cosine, separation
 * margin, recall@k vs PCA baseline) so in-browser / CI numbers are directly
 * comparable to offline Python numbers.
 *
 * Used by:
 *   - the runtime benchmark dashboard,
 *   - the CI fixture (T-027 gates),
 *   - the recall@10 SLO harness (T-012).
 */

export interface EmbeddingValidationReport {
  n: number;
  nClasses: number;
  embeddingDim: number;
  recallAtK: Record<string, number>;
  cosineAnalysis: {
    intraMean: number;
    intraStd: number;
    interMean: number;
    interStd: number;
    separationMargin: number;
    nIntraPairs: number;
    nInterPairs: number;
  };
  pcaBaseline: {
    pcaDim: number;
    recallAtK: number;
  };
  beatsPca: boolean;
  normMean: number;
  normStd: number;
  featureVarianceMean: number;
}

/** Cosine similarity matrix for a [N][D] embedding matrix. */
export function cosineMatrix(emb: number[][]): number[][] {
  const norms = emb.map((row) => Math.sqrt(row.reduce((s, v) => s + v * v, 0)) + 1e-9);
  const n = emb.length;
  const out: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let dot = 0;
      for (let d = 0; d < emb[i].length; d++) dot += emb[i][d] * emb[j][d];
      const sim = dot / (norms[i] * norms[j]);
      out[i][j] = sim;
      out[j][i] = sim;
    }
  }
  return out;
}

/** Recall@k: fraction of queries whose top-k nearest neighbours share a label. */
export function recallAtK(emb: number[][], labels: number[], k = 10): number {
  const n = emb.length;
  if (n === 0) return 0;
  const sim = cosineMatrix(emb);
  let hits = 0;
  for (let i = 0; i < n; i++) {
    // Exclude self by setting diagonal to -Infinity.
    sim[i][i] = -Infinity;
    const idx = argsortDesc(sim[i]).slice(0, k);
    if (idx.some((j) => labels[j] === labels[i])) hits++;
  }
  return hits / n;
}

/** Intra-class vs inter-class cosine similarity stats. */
export function intraInterClassCosine(
  emb: number[][],
  labels: number[],
): EmbeddingValidationReport["cosineAnalysis"] {
  const sim = cosineMatrix(emb);
  const n = emb.length;
  const intraSims: number[] = [];
  const interSims: number[] = [];

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (labels[i] === labels[j]) intraSims.push(sim[i][j]);
      else interSims.push(sim[i][j]);
    }
  }

  const intra = intraSims.length > 0 ? intraSims : [0];
  const inter = interSims.length > 0 ? interSims : [0];
  return {
    intraMean: mean(intra),
    intraStd: std(intra),
    interMean: mean(inter),
    interStd: std(inter),
    separationMargin: mean(intra) - mean(inter),
    nIntraPairs: intraSims.length,
    nInterPairs: interSims.length,
  };
}

/** PCA baseline: reduce to `targetDim` via SVD, then compute recall@k. */
export function pcaBaselineRecall(
  emb: number[][],
  labels: number[],
  k = 10,
  targetDim = 32,
): { pcaDim: number; recallAtK: number } {
  const reduced = pcaReduce(emb, Math.min(targetDim, emb[0]?.length ?? 1));
  return { pcaDim: reduced[0].length, recallAtK: recallAtK(reduced, labels, k) };
}

/** Full validation report. */
export function validateEmbeddings(
  emb: number[][],
  labels: number[],
  k = 10,
): EmbeddingValidationReport {
  const dim = emb[0]?.length ?? 0;
  const norms = emb.map((row) => Math.sqrt(row.reduce((s, v) => s + v * v, 0)));
  const recall = recallAtK(emb, labels, k);
  const pca = pcaBaselineRecall(emb, labels, k);
  const cosine = intraInterClassCosine(emb, labels);
  const nClasses = new Set(labels).size;

  return {
    n: emb.length,
    nClasses,
    embeddingDim: dim,
    recallAtK: { [k]: recall },
    cosineAnalysis: cosine,
    pcaBaseline: pca,
    beatsPca: recall > pca.recallAtK,
    normMean: mean(norms),
    normStd: std(norms),
    featureVarianceMean: mean(emb[0].map((_, d) => variance(emb.map((row) => row[d])))),
  };
}

// --- Helpers ---

function mean(arr: number[]): number {
  return arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr: number[]): number {
  if (arr.length === 0) return 0;
  const m = mean(arr);
  return Math.sqrt(mean(arr.map((v) => (v - m) * (v - m))));
}

function variance(arr: number[]): number {
  return std(arr) ** 2;
}

function argsortDesc(arr: number[]): number[] {
  return arr
    .map((v, i) => [v, i] as const)
    .sort((a, b) => b[0] - a[0])
    .map(([, i]) => i);
}

/** Simple SVD-based PCA reduction (power iteration fallback for robustness). */
function pcaReduce(emb: number[][], targetDim: number): number[][] {
  const n = emb.length;
  const d = emb[0].length;
  if (d <= targetDim) return emb;

  // Center.
  const colMeans = Array.from({ length: d }, (_, j) => mean(emb.map((row) => row[j])));
  const centered = emb.map((row) => row.map((v, j) => v - colMeans[j]));

  // Covariance matrix [d][d].
  const cov = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += centered[r][i] * centered[r][j];
      cov[i][j] = s / Math.max(1, n - 1);
    }
  }

  // Power iteration to find top-`targetDim` eigenvectors.
  const components: number[][] = [];
  for (let c = 0; c < targetDim; c++) {
    let v = Array.from({ length: d }, () => Math.random() - 0.5);
    let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    v = v.map((x) => x / (norm + 1e-12));
    for (let iter = 0; iter < 100; iter++) {
      const next = new Array<number>(d).fill(0);
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) next[i] += cov[i][j] * v[j];
      }
      norm = Math.sqrt(next.reduce((s, x) => s + x * x, 0));
      v = next.map((x) => x / (norm + 1e-12));
    }
    components.push(v);
    // Deflate: subtract the projection.
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        cov[i][j] -= v[i] * v[j] * cov[i][j]; // approximate deflation
      }
    }
  }

  // Project.
  return centered.map((row) =>
    components.map((comp) => row.reduce((s, x, j) => s + x * comp[j], 0)),
  );
}
