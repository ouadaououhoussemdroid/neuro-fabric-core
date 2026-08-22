/**
 * T-016 — Production readiness gate tests.
 *
 * Validates that all production-readiness requirements are met:
 *   1. All Tier-1 model artifacts exist in public/models/
 *   2. Manifest.json is valid JSON with all referenced models
 *   3. All required Supabase migrations exist
 *   4. All required CI jobs are present
 *   5. No secrets are committed to the repository
 *   6. No SSH keys are in the repository
 *   7. All Tier-1 services pull metrics from the registry (not hardcoded)
 */
import { describe, it, expect } from "vitest";
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";

const ROOT = process.cwd();
const MODELS_DIR = join(ROOT, "public", "models");
const MIGRATIONS_DIR = join(ROOT, "supabase", "migrations");

describe("Production Readiness — T-016", () => {
  describe("Tier-1 model artifacts", () => {
    const requiredModels = [
      "eegconformer.onnx",
      "cbramod-encoder.onnx",
      "eegpt-encoder-int8.onnx",
      join("cognitive", "cognitive-probe-joint2312-v1.onnx"),
      join("anomaly", "mahalanobis-probe-joint2312-v1.onnx"),
    ];

    for (const modelPath of requiredModels) {
      const fullPath = join(MODELS_DIR, modelPath);
      it(`model artifact exists: ${modelPath}`, () => {
        expect(existsSync(fullPath)).toBe(true);
      });

      it(`model artifact is non-trivial size: ${modelPath}`, () => {
        if (!existsSync(fullPath)) return; // skip if missing
        const stat = readFileSync(fullPath);
        expect(stat.length).toBeGreaterThan(100); // at least 100 bytes
      });
    }
  });

  describe("Manifest validation", () => {
    const manifestPath = join(MODELS_DIR, "manifest.json");

    it("manifest.json exists and is valid JSON", () => {
      expect(existsSync(manifestPath)).toBe(true);
      const content = readFileSync(manifestPath, "utf-8");
      const manifest = JSON.parse(content);
      expect(typeof manifest).toBe("object");
      expect(manifest).toHaveProperty("models");
    });

    it("manifest.json has entries for all Tier-1 probes", () => {
      const content = readFileSync(manifestPath, "utf-8");
      const manifest = JSON.parse(content);

      // Check for key model entries by their expected names
      const models = manifest.models || manifest;
      expect(models).toHaveProperty(["eegconformer"]);
      expect(models).toHaveProperty(["cognitive-probe-joint2312-v1"]);
      expect(models).toHaveProperty(["mahalanobis-probe-joint2312-v1"]);
    });
  });

  describe("Migration files", () => {
    it("has at least 20 migration files", () => {
      const files = readdirSync(MIGRATIONS_DIR).filter((f) => f.endsWith(".sql"));
      expect(files.length).toBeGreaterThanOrEqual(20);
    });

    it("includes joint_embeddings_2312 migration", () => {
      const files = readdirSync(MIGRATIONS_DIR);
      const migrationContent = files
        .filter((f) => f.endsWith(".sql"))
        .map((f) => readFileSync(join(MIGRATIONS_DIR, f), "utf-8"))
        .join("\n");
      expect(migrationContent).toContain("joint_embeddings_2312");
    });

    it("includes Tier-1 result tables in migrations", () => {
      const files = readdirSync(MIGRATIONS_DIR);
      const migrationContent = files
        .filter((f) => f.endsWith(".sql"))
        .map((f) => readFileSync(join(MIGRATIONS_DIR, f), "utf-8"))
        .join("\n");

      expect(migrationContent).toContain("subject_similarity_results");
      expect(migrationContent).toContain("cognitive_state_results");
      expect(migrationContent).toContain("anomaly_detection_results");
      expect(migrationContent).toContain("service_audit_log");
    });
  });

  describe("CI workflow", () => {
    const ciPath = join(ROOT, ".github", "workflows", "ci.yml");

    it("ci.yml exists", () => {
      expect(existsSync(ciPath)).toBe(true);
    });

    it("includes all required CI jobs", () => {
      const content = readFileSync(ciPath, "utf-8");
      expect(content).toMatch(/ci:/);
      expect(content).toMatch(/recall-slo:/);
      expect(content).toMatch(/security:/);
      expect(content).toMatch(/migration-validation:/);
      expect(content).toMatch(/browser-smoke:/);
      expect(content).toMatch(/native-inference:/);
    });

    it("includes native inference test step", () => {
      const content = readFileSync(ciPath, "utf-8");
      expect(content.toLowerCase()).toContain("onnxruntime-node");
    });

    it("does not have advisory || true on security audit", () => {
      const content = readFileSync(ciPath, "utf-8");
      // The security audit should not have `|| true` that silently passes
      const securitySection = content.split("security:")[1].split(/\n  [a-z]/)[0];
      expect(securitySection).not.toMatch(/bun audit.*\|\| true/);
    });
  });

  describe("No secrets committed", () => {
    it("no SSH private keys in repository root", () => {
      expect(existsSync(join(ROOT, "ENTER"))).toBe(false);
      expect(existsSync(join(ROOT, "ENTER.pub"))).toBe(false);
    });

    it("no sb_secret files in repository root", () => {
      const rootFiles = readdirSync(ROOT);
      const secretFiles = rootFiles.filter((f) => f.startsWith("sb_secret_"));
      expect(secretFiles).toHaveLength(0);
    });
  });

  describe("Tier-1 services use registry metrics", () => {
    const sleepServerPath = join(ROOT, "src", "lib", "ai", "inference", "sleep.server.ts");
    const cognitiveServerPath = join(ROOT, "src", "lib", "ai", "inference", "cognitive.server.ts");
    const anomalyServerPath = join(ROOT, "src", "lib", "ai", "inference", "anomaly.server.ts");

    it("sleep.server.ts pulls metrics from registry (not hardcoded zeros)", () => {
      const content = readFileSync(sleepServerPath, "utf-8");
      expect(content).toContain("SLEEP_STAGING_PROBE_JOINT_2312.training?.metrics");
      expect(content).toContain("SLEEP_QUALITY_PROBE_JOINT_2312.training?.metrics");
    });

    it("cognitive.server.ts pulls metrics from registry (not hardcoded)", () => {
      const content = readFileSync(cognitiveServerPath, "utf-8");
      expect(content).toContain("COGNITIVE_LINEAR_PROBE_JOINT_2312.training?.metrics");
      // Should NOT have hardcoded r2: 0.7348
      expect(content).not.toMatch(/r2:\s*0\.7348/);
    });

    it("anomaly.server.ts pulls metrics from registry", () => {
      const content = readFileSync(anomalyServerPath, "utf-8");
      expect(content).toContain("ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.training?.metrics");
    });
  });

  describe("Security hardening", () => {
    const clientPath = join(ROOT, "src", "integrations", "supabase", "client.ts");

    it("client.ts does not use localStorage for auth", () => {
      const content = readFileSync(clientPath, "utf-8");
      // Should not reference localStorage directly for auth storage
      expect(content).not.toMatch(/storage:\s*typeof window !== "undefined"\s*\?\s*localStorage/);
    });

    it("client.ts uses MemoryStorage class instead", () => {
      const content = readFileSync(clientPath, "utf-8");
      expect(content).toContain("MemoryStorage");
    });

    it("cookie-auth.server.ts exists", () => {
      const cookieAuthPath = join(ROOT, "src", "integrations", "supabase", "cookie-auth.server.ts");
      expect(existsSync(cookieAuthPath)).toBe(true);
    });

    it("WebSocket route requires authentication", () => {
      const wsPath = join(ROOT, "src", "routes", "api", "public", "stream", "$-source.ts");
      if (existsSync(wsPath)) {
        const content = readFileSync(wsPath, "utf-8");
        expect(content).toContain("authenticatePeer");
        expect(content).toContain("Unauthorized: missing token");
      }
    });
  });
});
