/**
 * T-014 — Subject-level embedding aggregation.
 *
 * Derives a per-subject signature from per-window 32-D embedding vectors,
 * with stability metrics across sessions. The aggregation strategy is the
 * **median** embedding (robust to outlier windows) plus an optional ICA
 * basis vector (extracted via PCA-based decomposition for now; Pyodide/ICA
 * reserved for the long-term path).
 *
 * Stability is quantified by:
 *   - `cosineSpread`: mean pairwise cosine distance among session signatures.
 *   - `normStd`: std of L2 norms across sessions.
 *   - `stabilityScore`: 1 - cosineSpread (higher = more stable).
 */

export interface WindowEmbedding {
  /** Per-window 32-D vector. */
  vector: number[];
  /** Session id this window belongs to. */
  sessionId: string;
}

export interface SessionSignature {
  sessionId: string;
  /** Median embedding across all windows in this session. */
  median: number[];
  /** Number of windows aggregated. */
  nWindows: number;
  /** L2 norm of the median. */
  norm: number;
}

export interface SubjectSignature {
  /** Per-session signatures. */
  sessions: SessionSignature[];
  /** Aggregated subject signature (median of session medians). */
  signature: number[];
  /** Number of sessions. */
  nSessions: number;
  /** Total windows across all sessions. */
  totalWindows: number;
  /** Stability metrics. */
  stability: {
    /** Mean pairwise cosine distance among session medians (0=identical, 2=opposite). */
    cosineSpread: number;
    /** Std of L2 norms across sessions. */
    normStd: number;
    /** 1 - cosineSpread (higher = more stable). */
    stabilityScore: number;
  };
  /** Dominant component direction (first PCA eigenvector of session medians). */
  dominantBasis: number[];
}

/**
 * Compute the element-wise median of a set of vectors.
 * Each output element is the median of the corresponding column.
 */
export function medianVector(vectors: number[][]): number[] {
  if (vectors.length === 0) return [];
  const dim = vectors[0].length;
  const out = new Array<number>(dim);
  for (let d = 0; d < dim; d++) {
    const col = vectors.map((v) => v[d]).sort((a, b) => a - b);
    const mid = Math.floor(col.length / 2);
    out[d] = col.length % 2 === 0 ? (col[mid - 1] + col[mid]) / 2 : col[mid];
  }
  return out;
}

/** L2 norm of a vector. */
export function l2Norm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

/** Cosine distance (1 - similarity) between two vectors. */
export function cosineDistance(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 1 : 1 - dot / denom;
}

/**
 * Compute a subject signature from per-window embeddings.
 *
 * 1. Group windows by session.
 * 2. For each session, compute the median embedding.
 * 3. The subject signature is the median of session medians.
 * 4. Stability metrics are computed from the session medians.
 */
export function aggregateSubjectSignature(windows: WindowEmbedding[]): SubjectSignature {
  if (windows.length === 0) {
    return {
      sessions: [],
      signature: [],
      nSessions: 0,
      totalWindows: 0,
      stability: { cosineSpread: 0, normStd: 0, stabilityScore: 1 },
      dominantBasis: [],
    };
  }

  // Group by session.
  const sessionMap = new Map<string, number[][]>();
  for (const w of windows) {
    if (!sessionMap.has(w.sessionId)) sessionMap.set(w.sessionId, []);
    sessionMap.get(w.sessionId)!.push(w.vector);
  }

  // Per-session medians.
  const sessions: SessionSignature[] = [];
  for (const [sessionId, vecs] of sessionMap) {
    const med = medianVector(vecs);
    sessions.push({ sessionId, median: med, nWindows: vecs.length, norm: l2Norm(med) });
  }

  // Subject signature = median of session medians.
  const signature = medianVector(sessions.map((s) => s.median));

  // Stability: pairwise cosine distance among session medians.
  const medians = sessions.map((s) => s.median);
  let sumDist = 0;
  let pairCount = 0;
  for (let i = 0; i < medians.length; i++) {
    for (let j = i + 1; j < medians.length; j++) {
      sumDist += cosineDistance(medians[i], medians[j]);
      pairCount++;
    }
  }
  const cosineSpread = pairCount > 0 ? sumDist / pairCount : 0;

  // Norm std across sessions.
  const norms = sessions.map((s) => s.norm);
  const normMean = norms.reduce((a, b) => a + b, 0) / Math.max(1, norms.length);
  const normStd = Math.sqrt(
    norms.reduce((s, n) => s + (n - normMean) ** 2, 0) / Math.max(1, norms.length),
  );

  // Dominant basis: first principal component via power iteration.
  const dominantBasis = firstPrincipalComponent(medians, signature);

  return {
    sessions,
    signature,
    nSessions: sessions.length,
    totalWindows: windows.length,
    stability: {
      cosineSpread,
      normStd,
      stabilityScore: 1 - cosineSpread,
    },
    dominantBasis,
  };
}

/**
 * First principal component via power iteration on centered session medians.
 * This is the "ICA basis" surrogate — a lightweight PCA approximation that
 * captures the dominant direction of inter-session variance. The full ICA
 * path (via Pyodide) is reserved for when non-Gaussian source separation
 * is needed.
 */
function firstPrincipalComponent(medians: number[][], center: number[]): number[] {
  if (medians.length < 2) return center.length > 0 ? center.slice() : [];
  const dim = medians[0].length;

  // Center.
  const centered = medians.map((m) => m.map((v, d) => v - center[d]));

  // Power iteration on the covariance.
  let v = new Array<number>(dim).fill(1 / Math.sqrt(dim));
  for (let iter = 0; iter < 50; iter++) {
    const next = new Array<number>(dim).fill(0);
    for (let i = 0; i < dim; i++) {
      for (const row of centered) {
        next[i] += row[i] * row.reduce((s, x, j) => s + x * v[j], 0);
      }
    }
    const norm = l2Norm(next);
    if (norm < 1e-12) break;
    v = next.map((x) => x / norm);
  }
  return v;
}
