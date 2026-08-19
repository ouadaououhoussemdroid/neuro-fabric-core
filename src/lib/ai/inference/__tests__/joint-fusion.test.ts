/**
 * M25 — Unit tests for the joint 264-D fusion function.
 *
 * Tests `fuseJointEmbedding` with synthetic inputs (no ONNX, no server runtime).
 * Mirrors the validation style of foundation-e2e.test.ts and the wasm-smoke
 * dim/normalization assertions.
 */
import { describe, it, expect } from "vitest";
import {
  fuseJointEmbedding,
  JOINT_EMBEDDING_DIM,
  JOINT_BLOCK_WEIGHTS,
  JOINT_COMPONENT_DIMS,
  JOINT_MODEL_ID,
} from "../joint.server";

/** Build a deterministic vector of length n with a constant baseline value. */
function vec(n: number, base = 0.01): number[] {
  return Array.from({ length: n }, () => base);
}

/** L2 norm of a vector. */
function l2Norm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

describe("M25 — Joint 264-D fusion", () => {
  describe("constants", () => {
    it("JOINT_EMBEDDING_DIM = 264", () => {
      expect(JOINT_EMBEDDING_DIM).toBe(264);
      const sum = JOINT_COMPONENT_DIMS.cbramod + JOINT_COMPONENT_DIMS.v2 + JOINT_COMPONENT_DIMS.pca;
      expect(sum).toBe(264);
    });

    it("block weights sum to 1.0", () => {
      const total = JOINT_BLOCK_WEIGHTS.cbramod + JOINT_BLOCK_WEIGHTS.v2 + JOINT_BLOCK_WEIGHTS.pca;
      expect(total).toBeCloseTo(1.0, 4);
    });

    it("block weights match M18 learned values: CBraMod=0.62, V2=0.16, PCA=0.22", () => {
      expect(JOINT_BLOCK_WEIGHTS.cbramod).toBe(0.62);
      expect(JOINT_BLOCK_WEIGHTS.v2).toBe(0.16);
      expect(JOINT_BLOCK_WEIGHTS.pca).toBe(0.22);
    });

    it("JOINT_MODEL_ID is the fused model id", () => {
      expect(JOINT_MODEL_ID).toBe("onnx-cbramod-joint-264");
    });
  });

  describe("fuseJointEmbedding", () => {
    it("produces exactly 264-D output from 200+32+32 inputs", () => {
      const result = fuseJointEmbedding(vec(200), vec(32), vec(32));
      expect(result).toHaveLength(264);
    });

    it("L2-normalises the output vector", () => {
      const result = fuseJointEmbedding(vec(200, 0.5), vec(32, 0.5), vec(32, 0.5));
      expect(l2Norm(result)).toBeCloseTo(1, 4);
    });

    it("zero-valued inputs produce all-zero output (L2 of zero → zero, not NaN)", () => {
      const result = fuseJointEmbedding(vec(200, 0), vec(32, 0), vec(32, 0));
      expect(result).toHaveLength(264);
      expect(result.every((v) => v === 0)).toBe(true);
      expect(result.some((v) => Number.isNaN(v))).toBe(false);
    });

    it("applies block weights correctly — only CBraMod active", () => {
      // CBraMod = all 1s (normalised to unit), V2 and PCA = all 0s.
      // After fusion, CBraMod block carries the entire norm.
      const result = fuseJointEmbedding(vec(200, 1), vec(32, 0), vec(32, 0));

      const cbNorm = l2Norm(result.slice(0, 200));
      const v2Norm = l2Norm(result.slice(200, 232));
      const pcaNorm = l2Norm(result.slice(232, 264));

      expect(v2Norm).toBeCloseTo(0, 10);
      expect(pcaNorm).toBeCloseTo(0, 10);
      expect(cbNorm).toBeCloseTo(1, 4);
    });

    it("preserves block weight energy ratio between two non-zero blocks", () => {
      // CBraMod = all 1s, V2 = all 1s, PCA = all 0s.
      // After per-block L2-normalise each block has energy = 1.0 (regardless of
      // element count). After block-weight scaling: CBraMod energy = 0.62²,
      // V2 energy = 0.16². After global L2-norm the ratio is preserved.
      const result = fuseJointEmbedding(vec(200, 1), vec(32, 1), vec(32, 0));

      const cbEnergy = result.slice(0, 200).reduce((s, v) => s + v * v, 0);
      const v2Energy = result.slice(200, 232).reduce((s, v) => s + v * v, 0);
      // After global L2-norm, total energy = 1, so cbEnergy + v2Energy ≈ 1
      const totalEnergy = cbEnergy + v2Energy;
      expect(totalEnergy).toBeCloseTo(1, 4);

      // Energy ratio = (0.62² × 1) / (0.16² × 1) = 0.62² / 0.16²
      const expectedRatio = JOINT_BLOCK_WEIGHTS.cbramod ** 2 / JOINT_BLOCK_WEIGHTS.v2 ** 2;
      expect(cbEnergy / v2Energy).toBeCloseTo(expectedRatio, 2);
    });

    it("throws on wrong CBraMod dimension", () => {
      expect(() => fuseJointEmbedding(vec(100), vec(32), vec(32))).toThrow(
        /CBraMod vector dim 100 != 200/,
      );
    });

    it("throws on wrong V2 dimension", () => {
      expect(() => fuseJointEmbedding(vec(200), vec(64), vec(32))).toThrow(
        /V2 vector dim 64 != 32/,
      );
    });

    it("throws on wrong PCA dimension", () => {
      expect(() => fuseJointEmbedding(vec(200), vec(32), vec(64))).toThrow(
        /PCA vector dim 64 != 32/,
      );
    });

    it("is deterministic — same inputs produce byte-identical output", () => {
      const cb = Array.from({ length: 200 }, () => (Math.random() - 0.5) * 2);
      const v2 = Array.from({ length: 32 }, () => (Math.random() - 0.5) * 2);
      const pca = Array.from({ length: 32 }, () => (Math.random() - 0.5) * 2);
      const r1 = fuseJointEmbedding(cb, v2, pca);
      const r2 = fuseJointEmbedding(cb, v2, pca);
      expect(r1).toEqual(r2);
    });

    it("cosine similarity of two runs = 1.0 (exact determinism)", () => {
      const cb = Array.from({ length: 200 }, () => (Math.random() - 0.5) * 2);
      const v2 = Array.from({ length: 32 }, () => (Math.random() - 0.5) * 2);
      const pca = Array.from({ length: 32 }, () => (Math.random() - 0.5) * 2);
      const r1 = fuseJointEmbedding(cb, v2, pca);
      const r2 = fuseJointEmbedding(cb, v2, pca);
      const cos = r1.reduce((s, v, i) => s + v * r2[i], 0);
      expect(cos).toBeCloseTo(1, 8);
    });

    it("handles random (non-degenerate) inputs and produces valid L2-normalised output", () => {
      const cb = Array.from({ length: 200 }, () => (Math.random() - 0.5) * 2);
      const v2 = Array.from({ length: 32 }, () => (Math.random() - 0.5) * 2);
      const pca = Array.from({ length: 32 }, () => (Math.random() - 0.5) * 2);
      const result = fuseJointEmbedding(cb, v2, pca);

      expect(result).toHaveLength(264);
      expect(l2Norm(result)).toBeCloseTo(1, 4);
      // No NaN or Inf
      for (const v of result) {
        expect(Number.isFinite(v)).toBe(true);
      }
      // Non-degenerate
      expect(l2Norm(result)).toBeGreaterThan(0.9);
    });
  });
});
