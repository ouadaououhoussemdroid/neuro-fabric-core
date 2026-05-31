/**
 * Streaming-friendly PCA via power iteration on the covariance matrix.
 * Works on small feature matrices (rows = samples, cols = features).
 * No external deps.
 */
export interface PCAModel {
  mean: number[];
  components: number[][]; // [k][d]
  explainedVariance: number[];
}

function matMulVec(A: number[][], x: number[]): number[] {
  const r = A.length;
  const out = new Array<number>(r).fill(0);
  for (let i = 0; i < r; i++) {
    const row = A[i];
    let s = 0;
    for (let j = 0; j < row.length; j++) s += row[j] * x[j];
    out[i] = s;
  }
  return out;
}

function norm(x: number[]): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s);
}

function normalize(x: number[]): number[] {
  const n = norm(x) || 1;
  return x.map((v) => v / n);
}

/** Fit PCA with the top-k components using power iteration with deflation. */
export function fitPCA(X: number[][], k: number, iters = 60): PCAModel {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  if (n === 0 || d === 0) throw new Error("PCA: empty input");

  const mean = new Array<number>(d).fill(0);
  for (const row of X) for (let j = 0; j < d; j++) mean[j] += row[j];
  for (let j = 0; j < d; j++) mean[j] /= n;

  // Centered matrix
  const Xc = X.map((row) => row.map((v, j) => v - mean[j]));

  // Covariance (d x d). For d up to a few hundred this is fine.
  const cov: number[][] = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  for (const row of Xc) {
    for (let i = 0; i < d; i++) {
      const ri = row[i];
      for (let j = i; j < d; j++) {
        cov[i][j] += ri * row[j];
      }
    }
  }
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      cov[i][j] /= n - 1 || 1;
      cov[j][i] = cov[i][j];
    }
  }

  const components: number[][] = [];
  const eigs: number[] = [];
  const A = cov.map((row) => row.slice());

  for (let comp = 0; comp < Math.min(k, d); comp++) {
    let v = new Array<number>(d).fill(0).map(() => Math.random() - 0.5);
    v = normalize(v);
    let lambda = 0;
    for (let it = 0; it < iters; it++) {
      const Av = matMulVec(A, v);
      lambda = norm(Av);
      if (lambda === 0) break;
      v = Av.map((x) => x / lambda);
    }
    components.push(v);
    eigs.push(lambda);
    // Deflate: A = A - lambda * v v^T
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        A[i][j] -= lambda * v[i] * v[j];
      }
    }
  }

  return { mean, components, explainedVariance: eigs };
}

/** Project a single feature vector into the PCA subspace. */
export function transformPCA(model: PCAModel, x: number[]): number[] {
  const d = model.mean.length;
  const centered = new Array<number>(d);
  for (let i = 0; i < d; i++) centered[i] = x[i] - model.mean[i];
  return model.components.map((c) => {
    let s = 0;
    for (let i = 0; i < d; i++) s += c[i] * centered[i];
    return s;
  });
}