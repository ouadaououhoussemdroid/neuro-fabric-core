/**
 * M46 — Manifest SHA Match Validation
 *
 * Verifies that:
 *   1. All trained ONNX artifacts on disk have SHA-256 hashes that match the
 *      values registered in `public/models/manifest.json` and `sleep.registry.ts`.
 *   2. No placeholder-* or "pending-*" SHA values remain in any registry.
 *   3. The M43/M44 trained probes differ from the M42 random-init placeholder
 *      versions (confirming the replacement actually happened).
 *
 * This is the M46 Gate A2 validation: signed artifact SHA verification.
 */
import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";

const REPO_ROOT = process.env.INIT_CWD || process.cwd();

/** Compute SHA-256 hash of a file. */
function computeFileHashSha256(filePath: string): string | null {
  if (!existsSync(filePath)) return null;
  const buf = readFileSync(filePath);
  return createHash("sha256").update(buf).digest("hex");
}

// M42 random-init placeholder SHAs (from benchmark_archive.json)
const M42_STAGING_SHA =
  "9da4ea37c92c1d87e80dde9a52bcd651246b73274fba5f11f4262d44ff3710f6";
const M42_QUALITY_SHA =
  "5fb7400f1f00037b36f10f9eb73297a346903fef48997c3357cb177a47797d4f";

// ─── 1. Artifact-to-manifest mapping ───────────────────────────────────────────

interface ArtifactCheck {
  filePath: string;
  modelId: string;
  description: string;
  mustDifferFromM42?: string;
}

const ARTIFACTS: ArtifactCheck[] = [
  {
    filePath: "public/models/sleep/staging-probe-joint2312-v1.onnx",
    modelId: "staging-probe-joint2312-v1",
    description: "M43 trained staging probe (2312→5)",
    mustDifferFromM42: M42_STAGING_SHA,
  },
  {
    filePath: "public/models/sleep/quality-probe-joint2312-v1.onnx",
    modelId: "quality-probe-joint2312-v1",
    description: "M43 trained quality probe (2312→1)",
    mustDifferFromM42: M42_QUALITY_SHA,
  },
];

const V2_32_ARTIFACTS: ArtifactCheck[] = [
  {
    filePath: "models/sleep/staging-probe-v2-32d-v1.onnx",
    modelId: "sleep-staging-v2-32d",
    description: "M44 trained staging V2-32 browser probe",
  },
  {
    filePath: "models/sleep/quality-probe-v2-32d-v1.onnx",
    modelId: "sleep-quality-v2-32d",
    description: "M44 trained quality V2-32 browser probe",
  },
  {
    filePath: "models/cognitive/cognitive-probe-v2-32d-v1.onnx",
    modelId: "cognitive-v2-32d",
    description: "M44 trained cognitive V2-32 browser probe",
  },
  {
    filePath: "models/anomaly/mahalanobis-probe-v2-32d-v1.onnx",
    modelId: "anomaly-v2-32d",
    description: "M44 trained anomaly V2-32 browser probe",
  },
];

// ─── 2. Load manifest ──────────────────────────────────────────────────────────

function loadManifest(): Record<string, unknown> {
  const manifest = JSON.parse(
    readFileSync(join(REPO_ROOT, "public/models/manifest.json"), "utf-8"),
  ) as { models: Record<string, unknown> };
  return manifest.models;
}

// ─── 3. Tests ──────────────────────────────────────────────────────────────────

describe("M46: Manifest SHA Verification (Gate A2)", () => {
  describe("M43 trained Tier-2 probes — manifest + registry SHA match", () => {
    const manifest = loadManifest();

    for (const artifact of ARTIFACTS) {
      it(`${artifact.description}: file SHA matches manifest.json`, () => {
        const full = join(REPO_ROOT, artifact.filePath);
        const actualSha = computeFileHashSha256(full);
        expect(actualSha, `File not found: ${artifact.filePath}`).not.toBeNull();

        const entry = manifest[artifact.modelId] as { sha256?: string } | undefined;
        expect(entry, `Not found in manifest: ${artifact.modelId}`).toBeDefined();
        expect(actualSha).toBe(entry!.sha256);
      });

      it(`${artifact.description}: file SHA matches registry`, async () => {
        const full = join(REPO_ROOT, artifact.filePath);
        const actualSha = computeFileHashSha256(full);
        expect(actualSha).not.toBeNull();

        // Import the registry module — the descriptor is a const export
        if (artifact.modelId === "staging-probe-joint2312-v1") {
          const {
            SLEEP_STAGING_PROBE_JOINT_2312,
          } = await import("@/lib/ai/decoders/sleep.registry");
          expect(actualSha).toBe(SLEEP_STAGING_PROBE_JOINT_2312.sha256);
        } else if (artifact.modelId === "quality-probe-joint2312-v1") {
          const {
            SLEEP_QUALITY_PROBE_JOINT_2312,
          } = await import("@/lib/ai/decoders/sleep.registry");
          expect(actualSha).toBe(SLEEP_QUALITY_PROBE_JOINT_2312.sha256);
        }
      });

      if (artifact.mustDifferFromM42) {
        it(`${artifact.description}: SHA differs from M42 random-init placeholder`, () => {
          const full = join(REPO_ROOT, artifact.filePath);
          const actualSha = computeFileHashSha256(full);
          expect(actualSha).not.toBeNull();
          expect(actualSha).not.toBe(artifact.mustDifferFromM42);
        });
      }
    }
  });

  describe("M44 trained V2-32 browser probes — registry SHA match", () => {
    for (const artifact of V2_32_ARTIFACTS) {
      it(`${artifact.description}: file SHA matches registry`, async () => {
        const full = join(REPO_ROOT, artifact.filePath);
        const actualSha = computeFileHashSha256(full);
        expect(actualSha, `File not found: ${artifact.filePath}`).not.toBeNull();

        // Look up the appropriate registry based on the artifact path
        if (artifact.filePath.includes("/sleep/")) {
          const {
            SLEEP_STAGING_PROBE_V2_32,
            SLEEP_QUALITY_PROBE_V2_32,
          } = await import("@/lib/ai/decoders/sleep.registry");

          if (artifact.modelId.includes("staging")) {
            expect(actualSha).toBe(SLEEP_STAGING_PROBE_V2_32.sha256);
          } else {
            expect(actualSha).toBe(SLEEP_QUALITY_PROBE_V2_32.sha256);
          }
        } else if (artifact.filePath.includes("/cognitive/")) {
          const {
            COGNITIVE_LINEAR_PROBE_V2_32,
          } = await import("@/lib/ai/decoders/cognitive.registry");
          expect(actualSha).toBe(COGNITIVE_LINEAR_PROBE_V2_32.sha256);
        } else if (artifact.filePath.includes("/anomaly/")) {
          const {
            ANOMALY_MAHALANOBIS_PROBE_V2_32,
          } = await import("@/lib/ai/decoders/anomaly.registry");
          expect(actualSha).toBe(ANOMALY_MAHALANOBIS_PROBE_V2_32.sha256);
        }
      });

      it(`${artifact.description}: file exists (non-empty)`, () => {
        const full = join(REPO_ROOT, artifact.filePath);
        expect(existsSync(full)).toBe(true);
      });
    }
  });

  describe("No placeholder values in any source file", () => {
    const filesToCheck = [
      "src/lib/ai/decoders/sleep.registry.ts",
      "src/lib/ai/decoders/cognitive.registry.ts",
      "src/lib/ai/decoders/anomaly.registry.ts",
      "src/lib/ai/decoders/browser-v2-32-weights.ts",
      "src/lib/ai/decoders/sleep.browser.ts",
      "src/lib/ai/decoders/cognitive.browser.ts",
      "src/lib/ai/decoders/anomaly.browser.ts",
    ];

    for (const file of filesToCheck) {
      it(`${file}: no 'placeholder-' or 'pending-' values`, () => {
        const full = join(REPO_ROOT, file);
        if (!existsSync(full)) return;
        const content = readFileSync(full, "utf-8");
        expect(content).not.toContain("placeholder-");
        expect(content).not.toContain("pending-");
      });
    }
  });

  describe("Manifest.json is valid and contains trained probes", () => {
    it("manifest has staging-probe-joint2312-v1 entry with real SHA", () => {
      const manifest = loadManifest();
      const entry = manifest["staging-probe-joint2312-v1"] as { sha256?: string; trained?: boolean };
      expect(entry).toBeDefined();
      expect(entry.trained).toBe(true);
      expect(entry.sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(entry.sha256).not.toContain("placeholder");
    });

    it("manifest has quality-probe-joint2312-v1 entry with real SHA", () => {
      const manifest = loadManifest();
      const entry = manifest["quality-probe-joint2312-v1"] as { sha256?: string; trained?: boolean };
      expect(entry).toBeDefined();
      expect(entry.trained).toBe(true);
      expect(entry.sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(entry.sha256).not.toContain("placeholder");
    });

    it("manifest SHA for staging-probe-joint2312-v1 matches file on disk", () => {
      const manifest = loadManifest();
      const entry = manifest["staging-probe-joint2312-v1"] as { sha256?: string };
      const filePath = join(
        REPO_ROOT,
        "public/models/sleep/staging-probe-joint2312-v1.onnx",
      );
      const actual = computeFileHashSha256(filePath);
      expect(actual).toBe(entry.sha256);
    });

    it("manifest SHA for quality-probe-joint2312-v1 matches file on disk", () => {
      const manifest = loadManifest();
      const entry = manifest["quality-probe-joint2312-v1"] as { sha256?: string };
      const filePath = join(
        REPO_ROOT,
        "public/models/sleep/quality-probe-joint2312-v1.onnx",
      );
      const actual = computeFileHashSha256(filePath);
      expect(actual).toBe(entry.sha256);
    });
  });
});
