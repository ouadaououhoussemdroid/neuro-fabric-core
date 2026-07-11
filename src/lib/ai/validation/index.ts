/**
 * Embedding validation + normalisation utilities. Cheap, deterministic, and
 * safe to call on every inference result so the rest of the stack never sees
 * NaN / wrong-dim / zero vectors silently.
 */
export interface ValidationOptions {
  expectedDim?: number;
  /** When true, reject the all-zero vector. Default true. */
  rejectZero?: boolean;
  /** Absolute element clamp; values outside trigger validation error. */
  maxAbs?: number;
}

export class EmbeddingValidationError extends Error {
  constructor(
    public readonly code: string,
    msg: string,
  ) {
    super(msg);
    this.name = "EmbeddingValidationError";
  }
}

export function validateEmbedding(vec: readonly number[], opts: ValidationOptions = {}): void {
  if (!Array.isArray(vec) && !ArrayBuffer.isView(vec as never)) {
    throw new EmbeddingValidationError("not_array", "embedding is not an array");
  }
  if (vec.length === 0) {
    throw new EmbeddingValidationError("empty", "embedding has length 0");
  }
  if (opts.expectedDim != null && vec.length !== opts.expectedDim) {
    throw new EmbeddingValidationError(
      "dim_mismatch",
      `expected dim ${opts.expectedDim}, got ${vec.length}`,
    );
  }
  let allZero = true;
  for (let i = 0; i < vec.length; i++) {
    const v = vec[i];
    if (!Number.isFinite(v)) {
      throw new EmbeddingValidationError("nan_or_inf", `non-finite at index ${i}`);
    }
    if (opts.maxAbs != null && Math.abs(v) > opts.maxAbs) {
      throw new EmbeddingValidationError(
        "out_of_range",
        `|v[${i}]|=${Math.abs(v)} exceeds maxAbs=${opts.maxAbs}`,
      );
    }
    if (v !== 0) allZero = false;
  }
  if ((opts.rejectZero ?? true) && allZero) {
    throw new EmbeddingValidationError("zero_vector", "embedding is all zero");
  }
}

/** L2-normalise in place-returning a new array. Zero-norm passes through. */
export function l2Normalize(vec: readonly number[]): number[] {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  const n = Math.sqrt(s);
  if (n === 0) return vec.slice();
  const out = new Array<number>(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / n;
  return out;
}

/** True when |‖v‖₂ − 1| < tol. */
export function isUnitNorm(vec: readonly number[], tol = 1e-3): boolean {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  return Math.abs(Math.sqrt(s) - 1) < tol;
}
