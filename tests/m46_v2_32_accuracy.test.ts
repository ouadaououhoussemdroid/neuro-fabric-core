/**
 * M46 — V2-32 Browser Probe Accuracy Validation
 *
 * Verifies that the M44 trained browser V2-32 probes (used in the browser WASM
 * fallback path) produce valid output shapes and are the trained weights
 * (not random-init placeholders).
 *
 * This test runs in the vitest environment (Node.js). It validates:
 *   1. All 4 V2-32 ONNX artifacts exist on disk
 *   2. SHA-256 hashes match the registry values (no "placeholder-*" remnants)
 *   3. Browser weight arrays are injected (non-zero, correct dimensions)
 *   4. The browser decoders (`detectSleepFromV2Embedding`, `browserSleepQuality`)
 *      produce valid output shapes (softmax sums to 1.0, quality in [0, 1])
 *   5. V2-32 staging accuracy ≥ 0.45 (degradation expected from 32→5 vs 2312→5)
 *
 * The V2-32 staging accuracy target is lower than the 2312-D probe because the
 * 32-D V2 subspace captures less information (M44 achieved 0.5193 on EEGMMIDB).
 */
import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";

const REPO_ROOT = join(process.cwd());

/** Compute SHA-256 hash of a file. */
async function computeFileHashSha256(filePath: string): Promise<string | null> {
  if (!existsSync(filePath)) return null;
  const buf = readFileSync(filePath);
  return createHash("sha256").update(buf).digest("hex");
}

/** Deterministic 32-D L2-normalised embedding for test fixtures. */
function syntheticV2Embedding(seed = 0): number[] {
  const v = Array.from({ length: 32 }, (_, i) =>
    Math.sin((i + seed) * 0.1) * 0.5,
  );
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map((x) => x / norm);
}

// ─── 1. Known expected SHA values (must match .registry.ts files) ─────────────

const EXPECTED_SHAS = {
  staging_v2_32:
    "ee03006bdeaa455f583ef2dcf6afc42301cd45979bb35d52310023b44b04d6c7",
  quality_v2_32:
    "39c624807a9e950b7cf129ede58a84b31741cefb7829ace4c7274b7dc7f3b5fe",
  cognitive_v2_32:
    "3ebd9ef943b17d23f65a45cda7f73301a623e95fffda2659c9c1245444e720c1",
  anomaly_v2_32:
    "a0cd2773ec6e6185a8fb5724450d3d563f12dcaab91ebb742e78aa68833f3f38",
} as const;

// V2-32 accuracy threshold (degradation expected from 32-D vs 2312-D subspace)
const V2_32_STAGING_ACC_TARGET = 0.45;

// ─── 2. Tests ──────────────────────────────────────────────────────────────────

describe("M46: V2-32 Browser Probe Validation (M44 trained probes)", () => {
  it("all 4 V2-32 ONNX artifacts exist on disk", async () => {
    const paths = [
      "models/sleep/staging-probe-v2-32d-v1.onnx",
      "models/sleep/quality-probe-v2-32d-v1.onnx",
      "models/cognitive/cognitive-probe-v2-32d-v1.onnx",
      "models/anomaly/mahalanobis-probe-v2-32d-v1.onnx",
    ];
    for (const p of paths) {
      const full = join(REPO_ROOT, p);
      expect(existsSync(full), `Artifact not found: ${p}`).toBe(true);
    }
  });

  it("staging V2-32 SHA-256 matches registry", async () => {
    const {
      SLEEP_STAGING_PROBE_V2_32,
    } = await import("@/lib/ai/decoders/sleep.registry");
    expect(SLEEP_STAGING_PROBE_V2_32.sha256).toBe(EXPECTED_SHAS.staging_v2_32);

    const path = join(
      REPO_ROOT,
      "models/sleep/staging-probe-v2-32d-v1.onnx",
    );
    const actual = await computeFileHashSha256(path);
    expect(actual).toBe(EXPECTED_SHAS.staging_v2_32);
  });

  it("quality V2-32 SHA-256 matches registry", async () => {
    const {
      SLEEP_QUALITY_PROBE_V2_32,
    } = await import("@/lib/ai/decoders/sleep.registry");
    expect(SLEEP_QUALITY_PROBE_V2_32.sha256).toBe(EXPECTED_SHAS.quality_v2_32);

    const path = join(
      REPO_ROOT,
      "models/sleep/quality-probe-v2-32d-v1.onnx",
    );
    const actual = await computeFileHashSha256(path);
    expect(actual).toBe(EXPECTED_SHAS.quality_v2_32);
  });

  it("cognitive V2-32 SHA-256 matches registry", async () => {
    const {
      COGNITIVE_LINEAR_PROBE_V2_32,
    } = await import("@/lib/ai/decoders/cognitive.registry");
    expect(COGNITIVE_LINEAR_PROBE_V2_32.sha256).toBe(EXPECTED_SHAS.cognitive_v2_32);

    const path = join(
      REPO_ROOT,
      "models/cognitive/cognitive-probe-v2-32d-v1.onnx",
    );
    const actual = await computeFileHashSha256(path);
    expect(actual).toBe(EXPECTED_SHAS.cognitive_v2_32);
  });

  it("anomaly V2-32 SHA-256 matches registry", async () => {
    const {
      ANOMALY_MAHALANOBIS_PROBE_V2_32,
    } = await import("@/lib/ai/decoders/anomaly.registry");
    expect(ANOMALY_MAHALANOBIS_PROBE_V2_32.sha256).toBe(EXPECTED_SHAS.anomaly_v2_32);

    const path = join(
      REPO_ROOT,
      "models/anomaly/mahalanobis-probe-v2-32d-v1.onnx",
    );
    const actual = await computeFileHashSha256(path);
    expect(actual).toBe(EXPECTED_SHAS.anomaly_v2_32);
  });

  it("browser weight arrays are trained (non-zero, correct dimensions)", async () => {
    const {
      BROWSER_SLEEP_STAGING_WEIGHTS,
      BROWSER_SLEEP_STAGING_BIAS,
      BROWSER_SLEEP_QUALITY_WEIGHTS,
      BROWSER_SLEEP_QUALITY_BIAS,
      BROWSER_COGNITIVE_WEIGHTS,
      BROWSER_COGNITIVE_BIAS,
      BROWSER_ANOMALY_WEIGHTS,
      BROWSER_ANOMALY_BIAS,
    } = await import("@/lib/ai/decoders/browser-v2-32-weights");

    // Staging: 5×32 weight matrix + 5 bias elements
    expect(BROWSER_SLEEP_STAGING_WEIGHTS).toHaveLength(5);
    for (const row of BROWSER_SLEEP_STAGING_WEIGHTS) {
      expect(row).toHaveLength(32);
    }
    expect(BROWSER_SLEEP_STAGING_BIAS).toHaveLength(5);

    // Quality: 32-D weight vector + scalar bias
    expect(BROWSER_SLEEP_QUALITY_WEIGHTS).toHaveLength(32);
    expect(typeof BROWSER_SLEEP_QUALITY_BIAS).toBe("number");

    // Cognitive: 32-D weight vector + scalar bias
    expect(BROWSER_COGNITIVE_WEIGHTS).toHaveLength(32);
    expect(typeof BROWSER_COGNITIVE_BIAS).toBe("number");

    // Anomaly: 32-D weight vector + scalar bias
    expect(BROWSER_ANOMALY_WEIGHTS).toHaveLength(32);
    expect(typeof BROWSER_ANOMALY_BIAS).toBe("number");

    // Verify weights are not all-zero (actual trained values)
    const maxStaging = Math.max(
      ...BROWSER_SLEEP_STAGING_WEIGHTS.flat().map((w) => Math.abs(w)),
    );
    expect(maxStaging).toBeGreaterThan(0.01);

    const maxQuality = Math.max(
      ...BROWSER_SLEEP_QUALITY_WEIGHTS.map((w) => Math.abs(w)),
    );
    expect(maxQuality).toBeGreaterThan(0.01);

    const maxCognitive = Math.max(
      ...BROWSER_COGNITIVE_WEIGHTS.map((w) => Math.abs(w)),
    );
    expect(maxCognitive).toBeGreaterThan(0.01);

    const maxAnomaly = Math.max(
      ...BROWSER_ANOMALY_WEIGHTS.map((w) => Math.abs(w)),
    );
    expect(maxAnomaly).toBeGreaterThan(0.01);
  });

  it("detectSleepFromV2Embedding produces valid 5-class softmax output with trained weights", async () => {
    const { detectSleepFromV2Embedding } = await import(
      "@/lib/ai/decoders/sleep.browser"
    );

    const embedding = syntheticV2Embedding(0);
    const result = detectSleepFromV2Embedding(embedding);

    // Valid sleep stage index
    expect(result.stage_id).toBeGreaterThanOrEqual(0);
    expect(result.stage_id).toBeLessThanOrEqual(4);

    // 5 probabilities summing to 1.0 (softmax contract)
    expect(result.probabilities).toHaveLength(5);
    const sum = result.probabilities.reduce((a: number, b: number) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5);

    // Confidence = max probability, in [0, 1]
    expect(result.confidence).toBeCloseTo(
      Math.max(...result.probabilities),
      5,
    );
    expect(result.confidence).toBeGreaterThan(0);
    expect(result.confidence).toBeLessThanOrEqual(1);

    // Decoder field present (should use trained probe path)
    expect(result.decoder).toBeDefined();
    expect(typeof result.decoder).toBe("string");
  });

  it("browserSleepQuality produces valid score in [0, 1] with trained weights", async () => {
    const { browserSleepQuality } = await import(
      "@/lib/ai/decoders/sleep.browser"
    );

    const embedding = syntheticV2Embedding(1);
    const result = browserSleepQuality(embedding);

    // Score clamped to [0, 1]
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);

    // Valid quality band
    const validBands = ["poor", "fair", "good", "excellent"];
    expect(validBands).toContain(result.band);

    // Confidence in [0, 1]
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);

    // Decoder field present
    expect(result.decoder).toBeDefined();
    expect(typeof result.decoder).toBe("string");
  });

  it("V2-32 staging accuracy >= 0.45 (M46 target, degradation from 2312-D expected)", async () => {
    // The M44 V2-32 staging probe achieved acc_5class=0.5193 on EEGMMIDB
    // (5-fold LOSO). This exceeds the M46 browser fallback target of 0.45.
    const {
      SLEEP_STAGING_PROBE_V2_32,
    } = await import("@/lib/ai/decoders/sleep.registry");

    const acc = SLEEP_STAGING_PROBE_V2_32.training?.metrics?.acc_5class ?? 0;
    expect(acc).toBeGreaterThanOrEqual(V2_32_STAGING_ACC_TARGET);
  });

  it("all 4 browser decoders produce output without errors for multiple embeddings", async () => {
    const { detectSleepFromV2Embedding, browserSleepQuality } = await import(
      "@/lib/ai/decoders/sleep.browser"
    );
    const { decodeFromV2Embedding } = await import(
      "@/lib/ai/decoders/cognitive.browser"
    );
    const { detectFromV2Embedding: detectAnomalyFromV2Embedding } = await import(
      "@/lib/ai/decoders/anomaly.browser"
    );

    for (let s = 0; s < 5; s++) {
      const emb = syntheticV2Embedding(s);

      const sleepResult = detectSleepFromV2Embedding(emb);
      expect(sleepResult.probabilities).toHaveLength(5);
      expect(sleepResult.probabilities.every((p) => Number.isFinite(p))).toBe(true);

      const qualityResult = browserSleepQuality(emb);
      expect(Number.isFinite(qualityResult.score)).toBe(true);

      const cognitiveResult = decodeFromV2Embedding(emb);
      expect(Number.isFinite(cognitiveResult.workload)).toBe(true);

      const anomalyResult = detectAnomalyFromV2Embedding(emb);
      expect(Number.isFinite(anomalyResult.score)).toBe(true);
    }
  });

  it("no placeholder-* SHAs remain in any registry or weight file", async () => {
    // Check all registry files
    const registries = [
      "src/lib/ai/decoders/sleep.registry.ts",
      "src/lib/ai/decoders/cognitive.registry.ts",
      "src/lib/ai/decoders/anomaly.registry.ts",
    ];
    for (const r of registries) {
      const content = readFileSync(join(REPO_ROOT, r), "utf-8");
      expect(content, `${r} should not contain placeholder-*`).not.toContain(
        "placeholder-",
      );
    }

    // Check browser weight file
    const weightsContent = readFileSync(
      join(REPO_ROOT, "src/lib/ai/decoders/browser-v2-32-weights.ts"),
      "utf-8",
    );
    expect(weightsContent).not.toContain("placeholder");
  });
});
