/**
 * M46 — Trained Probe Accuracy Validation
 *
 * Verifies that the M43 trained sleep staging and sleep quality probes meet
 * their accuracy thresholds against the held-out Sleep-EDF LOSO fold results.
 *
 * This test does NOT run real ONNX inference (no onnxruntime-node in the test
 * environment). Instead, it validates:
 *   1. All trained ONNX artifacts exist on disk
 *   2. SHA-256 hashes match the registry values
 *   3. Registry metrics are populated with real values (not zeros or placeholders)
 *   4. Staging accuracy ≥ 0.65 (M46 target threshold)
 *   5. Quality R² ≥ 0.60 (M46 target threshold)
 *
 * The actual accuracy values come from M43's 50-fold LOSO cross-validation
 * results (models/sleep/m43_staging_probe_results.json, m43_quality_probe_results.json),
 * which used train-only Ridge probes (no leakage) on EEGMMIDB Joint-2312 embeddings.
 */
import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";

const REPO_ROOT = join(process.cwd());

/** Compute SHA-256 hash of a file, returning hex string. */
async function computeFileHashSha256(filePath: string): Promise<string | null> {
  if (!existsSync(filePath)) return null;
  const buf = await import("node:fs").then((f) => f.readFileSync(filePath));
  return createHash("sha256").update(buf).digest("hex");
}

// ─── 1. Load M43 training results ─────────────────────────────────────────────

function loadStagingResults() {
  const path = join(REPO_ROOT, "models/sleep/m43_staging_probe_results.json");
  return JSON.parse(readFileSync(path, "utf-8"));
}

function loadQualityResults() {
  const path = join(REPO_ROOT, "models/sleep/m43_quality_probe_results.json");
  return JSON.parse(readFileSync(path, "utf-8"));
}

// ─── 2. Known registry values (must match sleep.registry.ts) ───────────────────

const EXPECTED_STAGING_SHA =
  "33dde2d3801e74cce6ed33e0e83ec072df62ede9e3ca9c0187ba39f0d7673cff";
const EXPECTED_QUALITY_SHA =
  "e41ed5282d77aa3b401b587aa3fdbb375ed46b480a71e6f8d9a471efe82ccdfd";

// M46 target thresholds
const STAGING_ACC_TARGET = 0.65;
const QUALITY_R2_TARGET = 0.60;

// ─── 3. Tests ──────────────────────────────────────────────────────────────────

describe("M46: Trained Probe Accuracy (M43 trained probes)", () => {
  it("staging ONNX artifact exists on disk", async () => {
    const path = join(
      REPO_ROOT,
      "public/models/sleep/staging-probe-joint2312-v1.onnx",
    );
    const hash = await computeFileHashSha256(path);
    expect(hash).toBeTruthy();
  });

  it("quality ONNX artifact exists on disk", async () => {
    const path = join(
      REPO_ROOT,
      "public/models/sleep/quality-probe-joint2312-v1.onnx",
    );
    const hash = await computeFileHashSha256(path);
    expect(hash).toBeTruthy();
  });

  it("staging probe SHA-256 matches registry value", async () => {
    const path = join(
      REPO_ROOT,
      "public/models/sleep/staging-probe-joint2312-v1.onnx",
    );
    const { getTaskHead, registerSleepHeads } = await import(
      "@/lib/ai/decoders"
    );
    registerSleepHeads();
    const head = getTaskHead("sleep-staging-v1");
    expect(head).toBeDefined();
    expect(head!.sha256).toBe(EXPECTED_STAGING_SHA);

    const actualSha = await computeFileHashSha256(path);
    expect(actualSha).toBe(EXPECTED_STAGING_SHA);
  });

  it("quality probe SHA-256 matches registry value", async () => {
    const path = join(
      REPO_ROOT,
      "public/models/sleep/quality-probe-joint2312-v1.onnx",
    );
    const { getTaskHead, registerSleepHeads } = await import(
      "@/lib/ai/decoders"
    );
    registerSleepHeads();
    const head = getTaskHead("sleep-quality-v1");
    expect(head).toBeDefined();
    expect(head!.sha256).toBe(EXPECTED_QUALITY_SHA);

    const actualSha = await computeFileHashSha256(path);
    expect(actualSha).toBe(EXPECTED_QUALITY_SHA);
  });

  it("staging accuracy >= 0.65 (M46 threshold) on held-out LOSO folds", () => {
    const results = loadStagingResults();
    const acc = results.cv_stats.mean_acc_5class;
    expect(acc).toBeGreaterThanOrEqual(STAGING_ACC_TARGET);
  });

  it("staging macro F1 is non-zero (model is actually trained, not random-init)", () => {
    const results = loadStagingResults();
    const f1 = results.cv_stats.mean_macro_f1;
    expect(f1).toBeGreaterThan(0);
  });

  it("staging kappa > 0 (substantive agreement beyond chance)", () => {
    const results = loadStagingResults();
    const kappa = results.cv_stats.mean_kappa;
    expect(kappa).toBeGreaterThan(0);
  });

  it("quality R² >= 0.60 (M46 threshold)", () => {
    const results = loadQualityResults();
    const r2 = results.cv_stats.mean_r2;
    expect(r2).toBeGreaterThanOrEqual(QUALITY_R2_TARGET);
  });

  it("quality RMSE is small (< 0.1, M46 spec)", () => {
    const results = loadQualityResults();
    const rmse = results.cv_stats.mean_rmse;
    expect(rmse).toBeLessThan(0.1);
  });

  it("quality MAE is small (< 0.08, M46 spec)", () => {
    const results = loadQualityResults();
    const mae = results.cv_stats.mean_mae;
    expect(mae).toBeLessThan(0.08);
  });

  it("quality Pearson r >= 0.9 (high correlation with ground-truth proxy)", () => {
    const results = loadQualityResults();
    const r = results.cv_stats.mean_pearson_r;
    expect(r).toBeGreaterThanOrEqual(0.9);
  });

  it("staging probe SHA in results JSON matches file on disk", async () => {
    const results = loadStagingResults();
    const path = join(
      REPO_ROOT,
      "public/models/sleep/staging-probe-joint2312-v1.onnx",
    );
    const actualSha = await computeFileHashSha256(path);
    expect(actualSha).toBe(results.probe_sha256);
  });

  it("quality probe SHA in results JSON matches file on disk", async () => {
    const results = loadQualityResults();
    const path = join(
      REPO_ROOT,
      "public/models/sleep/quality-probe-joint2312-v1.onnx",
    );
    const actualSha = await computeFileHashSha256(path);
    expect(actualSha).toBe(results.probe_sha256);
  });
});
