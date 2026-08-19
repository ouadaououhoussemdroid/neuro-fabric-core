/**
 * M28 — Unit tests for the Joint-2312 4-block fusion function.
 *
 * Tests `fuseJoint2312Embedding` with synthetic inputs (no ONNX, no server runtime),
 * mirroring the validation style of `joint-fusion.test.ts` and the wasm-smoke
 * dim/normalization assertions.
 */
import { describe, it, expect } from "vitest";
import {
  fuseJoint2312Embedding,
  JOINT_2312_EMBEDDING_DIM,
  JOINT_2312_BLOCK_WEIGHTS,
  JOINT_2312_COMPONENT_DIMS,
  JOINT_2312_MODEL_ID,
} from "../joint.server";

/** Build a deterministic vector of length n with a constant baseline value. */
function vec(n: number, base = 0.01): number[] {
  return Array.from({ length: n }, () => base);
}

/** L2 norm of a vector. */
function l2Norm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

describe("M28 — Joint-2312 4-block fusion", () => {
  describe("constants", () => {
    it("JOINT_2312_EMBEDDING_DIM = 2312", () => {
      expect(JOINT_2312_EMBEDDING_DIM).toBe(2312);
      const sum =
        JOINT_2312_COMPONENT_DIMS.cbramod +
        JOINT_2312_COMPONENT_DIMS.v2 +
        JOINT_2312_COMPONENT_DIMS.pca +
        JOINT_2312_COMPONENT_DIMS.eegpt;
      expect(sum).toBe(2312);
    });

    it("block weights sum to 1.0", () => {
      const total =
        JOINT_2312_BLOCK_WEIGHTS.cbramod +
        JOINT_2312_BLOCK_WEIGHTS.v2 +
        JOINT_2312_BLOCK_WEIGHTS.pca +
        JOINT_2312_BLOCK_WEIGHTS.eegpt;
      expect(total).toBeCloseTo(1.0, 3);
    });

    it("block weights match M27 learned values", () => {
      expect(JOINT_2312_BLOCK_WEIGHTS.cbramod).toBeCloseTo(0.3062, 3);
      expect(JOINT_2312_BLOCK_WEIGHTS.v2).toBeCloseTo(0.1434, 3);
      expect(JOINT_2312_BLOCK_WEIGHTS.pca).toBeCloseTo(0.1519, 3);
      expect(JOINT_2312_BLOCK_WEIGHTS.eegpt).toBeCloseTo(0.3985, 3);
    });

    it("JOINT_2312_MODEL_ID is the fused model id", () => {
      expect(JOINT_2312_MODEL_ID).toBe("onnx-cbramod-joint-2312");
    });
  });

  describe("fuseJoint2312Embedding", () => {
    it("produces exactly 2312-D output from 200+32+32+2048 inputs", () => {
      const result = fuseJoint2312Embedding(vec(200), vec(32), vec(32), vec(2048));
      expect(result).toHaveLength(2312);
    });

    it("L2-normalises the output vector", () => {
      const result = fuseJoint2312Embedding(vec(200, 0.5), vec(32, 0.5), vec(32, 0.5), vec(2048, 0.5));
      expect(l2Norm(result)).toBeCloseTo(1, 4);
    });

    it("zero-valued inputs produce all-zero output (L2 of zero → zero, not NaN)", () => {
      const result = fuseJoint2312Embedding(vec(200, 0), vec(32, 0), vec(32, 0), vec(2048, 0));
      expect(result).toHaveLength(2312);
      expect(result.every((v) => v === 0)).toBe(true);
      expect(result.some((v) => Number.isNaN(v))).toBe(false);
    });

    it("throws on wrong CBraMod dimension", () => {
      expect(() => fuseJoint2312Embedding(vec(100), vec(32), vec(32), vec(2048))).toThrow(
        /CBraMod vector dim 100 != 200/,
      );
    });

    it("throws on wrong V2 dimension", () => {
      expect(() => fuseJoint2312Embedding(vec(200), vec(64), vec(32), vec(2048))).toThrow(
        /V2 vector dim 64 != 32/,
      );
    });

    it("throws on wrong PCA dimension", () => {
      expect(() => fuseJoint2312Embedding(vec(200), vec(32), vec(64), vec(2048))).toThrow(
        /PCA vector dim 64 != 32/,
      );
    });

    it("throws on wrong EEGPT dimension", () => {
      expect(() => fuseJoint2312Embedding(vec(200), vec(32), vec(32), vec(1024))).toThrow(
        /EEGPT vector dim 1024 != 2048/,
      );
    });

    it("applies block weights correctly — only CBraMod active", () => {
      const result = fuseJoint2312Embedding(vec(200, 1), vec(32, 0), vec(32, 0), vec(2048, 0));

      const cbNorm = l2Norm(result.slice(0, 200));
      const v2Norm = l2Norm(result.slice(200, 232));
      const pcaNorm = l2Norm(result.slice(232, 264));
      const eegptNorm = l2Norm(result.slice(264, 2312));

      expect(v2Norm).toBeCloseTo(0, 10);
      expect(pcaNorm).toBeCloseTo(0, 10);
      expect(eegptNorm).toBeCloseTo(0, 10);
      expect(cbNorm).toBeCloseTo(1, 4);
    });

    it("preserves block weight energy ratio between two non-zero blocks", () => {
      const result = fuseJoint2312Embedding(vec(200, 1), vec(32, 1), vec(32, 0), vec(2048, 0));

      const cbEnergy = result.slice(0, 200).reduce((s, v) => s + v * v, 0);
      const v2Energy = result.slice(200, 232).reduce((s, v) => s + v * v, 0);
      const totalEnergy = cbEnergy + v2Energy;
      expect(totalEnergy).toBeCloseTo(1, 4);

      const expectedRatio = JOINT_2312_BLOCK_WEIGHTS.cbramod ** 2 / JOINT_2312_BLOCK_WEIGHTS.v2 ** 2;
      expect(cbEnergy / v2Energy).toBeCloseTo(expectedRatio, 2);
    });

    it("preserves energy ratio including EEGPT block", () => {
      const result = fuseJoint2312Embedding(vec(200, 1), vec(32, 0), vec(32, 0), vec(2048, 1));

      const cbEnergy = result.slice(0, 200).reduce((s, v) => s + v * v, 0);
      const eegptEnergy = result.slice(264, 2312).reduce((s, v) => s + v * v, 0);
      const totalEnergy = cbEnergy + eegptEnergy;
      expect(totalEnergy).toBeCloseTo(1, 4);

      const expectedRatio =
        JOINT_2312_BLOCK_WEIGHTS.cbramod ** 2 / JOINT_2312_BLOCK_WEIGHTS.eegpt ** 2;
      expect(cbEnergy / eegptEnergy).toBeCloseTo(expectedRatio, 4);
    });

    it("is deterministic — same inputs produce byte-identical output", () => {
      const cb = Array.from({ length: 200 }, () => (Math.random() - 0.5) * 2);
      const v2 = Array.from({ length: 32 }, () => (Math.random() - 0.5) * 2);
      const pca = Array.from({ length: 32 }, () => (Math.random() - 0.5) * 2);
      const eegpt = Array.from({ length: 2048 }, () => (Math.random() - 0.5) * 2);
      const r1 = fuseJoint2312Embedding(cb, v2, pca, eegpt);
      const r2 = fuseJoint2312Embedding(cb, v2, pca, eegpt);
      expect(r1).toEqual(r2);
    });

    it("cosine similarity of two runs = 1.0 (exact determinism)", () => {
      const cb = Array.from({ length: 200 }, () => (Math.random() - 0.5) * 2);
      const v2 = Array.from({ length: 32 }, () => (Math.random() - 0.5) * 2);
      const pca = Array.from({ length: 32 }, () => (Math.random() - 0.5) * 2);
      const eegpt = Array.from({ length: 2048 }, () => (Math.random() - 0.5) * 2);
      const r1 = fuseJoint2312Embedding(cb, v2, pca, eegpt);
      const r2 = fuseJoint2312Embedding(cb, v2, pca, eegpt);
      const cos = r1.reduce((s, v, i) => s + v * r2[i], 0);
      expect(cos).toBeCloseTo(1, 8);
    });

    it("handles random (non-degenerate) inputs and produces valid L2-normalised output", () => {
      const cb = Array.from({ length: 200 }, () => (Math.random() - 0.5) * 2);
      const v2 = Array.from({ length: 32 }, () => (Math.random() - 0.5) * 2);
      const pca = Array.from({ length: 32 }, () => (Math.random() - 0.5) * 2);
      const eegpt = Array.from({ length: 2048 }, () => (Math.random() - 0.5) * 2);
      const result = fuseJoint2312Embedding(cb, v2, pca, eegpt);

      expect(result).toHaveLength(2312);
      expect(l2Norm(result)).toBeCloseTo(1, 4);
      for (const v of result) {
        expect(Number.isFinite(v)).toBe(true);
      }
      expect(l2Norm(result)).toBeGreaterThan(0.9);
    });
  });
});
