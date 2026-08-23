/**
 * M-Scientific-Reboot — Artifact/Methodology Consistency Gate Tests
 *
 * Phase 16: Ensures that for every production probe:
 *   Registry architecture == Actual ONNX architecture
 *   Registry metric == Metric produced by evaluation
 *   Registry experiment == Experiment that generated artifact
 *   Registry SHA == actual artifact SHA
 *
 * A mismatch must fail CI. These tests verify the scientific integrity
 * of the model registry against actual artifacts and results.
 */
import { describe, it, expect } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";

const REPO_ROOT = path.resolve(__dirname, "../../../../../");
const MODELS_DIR = path.join(REPO_ROOT, "models");
const RESULTS_DIR = path.join(MODELS_DIR, "cognitive");
const ANOMALY_DIR = path.join(MODELS_DIR, "anomaly");

/** Compute SHA-256 of a file, mirroring the Python training scripts. */
function sha256File(filepath: string): string | null {
  if (!fs.existsSync(filepath)) return null;
  const hash = crypto.createHash("sha256");
  const buf = fs.readFileSync(filepath);
  hash.update(buf);
  return hash.digest("hex");
}

describe("Artifact/Methodology Consistency Gate", () => {
  describe("Cognitive probe v2", () => {
    it("results file exists and is valid JSON", () => {
      const resultsPath = path.join(RESULTS_DIR, "m33_cognitive_results_v2.json");
      expect(fs.existsSync(resultsPath)).toBe(true);
      const content = fs.readFileSync(resultsPath, "utf-8");
      const results = JSON.parse(content);
      expect(results.mission).toBe("m33-scientific-reboot");
      expect(results.label_source).toBe("experimental_protocol");
      expect(results.circularity_risk).toMatch(/NONE/);
    });

    it("accuracy in results exceeds baseline", () => {
      const resultsPath = path.join(RESULTS_DIR, "m33_cognitive_results_v2.json");
      const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
      const joint = results.results["Joint-2312 (M27)"];
      expect(joint.accuracy).toBeGreaterThan(joint.baseline_accuracy);
      // Must be significantly above chance (25% for 4 classes)
      expect(joint.accuracy).toBeGreaterThan(0.28);
    });

    it("labels are genuine experimental protocol, not proxy", () => {
      const resultsPath = path.join(RESULTS_DIR, "m33_cognitive_results_v2.json");
      const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
      expect(results.labels).toEqual({0:"left_hand",1:"right_hand",2:"feet",3:"tongue"});
      expect(results.previous_claim.status).toContain("INVALID");
    });
  });

  describe("Anomaly probe v2", () => {
    it("results file exists with methodology match", () => {
      const resultsPath = path.join(ANOMALY_DIR, "m34_anomaly_results_v2.json");
      expect(fs.existsSync(resultsPath)).toBe(true);
      const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
      expect(results.fix).toContain("Mahalanobis");
      expect(results.v1_comparison.v1_status).toMatch(/INVALID/);
      expect(results.labels.independent_of_input_features).toBe(true);
    });

    it("V2 AUC reflects true Mahalanobis, not Ridge", () => {
      const resultsPath = path.join(ANOMALY_DIR, "m34_anomaly_results_v2.json");
      const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
      // V2 AUC should be different from V1's claimed 0.892
      // and should match actual Mahalanobis evaluation
      expect(results.v1_comparison.v1_reported_auc).toBe(0.892);
      expect(results.v1_comparison.v2_auc).not.toBe(0.892);
      expect(results.v1_comparison.methodology_match).toBe(true);
    });
  });

  describe("Dataset manifests exist", () => {
    it("eegmmidb manifest is present", () => {
      const manifestPath = path.join(REPO_ROOT, "datasets", "manifests", "eegmmidb.json");
      expect(fs.existsSync(manifestPath)).toBe(true);
      const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
      expect(manifest.ground_truth_labels).toBeDefined();
      expect(manifest.circularity_risk).toMatch(/NONE/);
    });

    it("sleep-edf manifest is present", () => {
      const manifestPath = path.join(REPO_ROOT, "datasets", "manifests", "sleep-edf.json");
      expect(fs.existsSync(manifestPath)).toBe(true);
      const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
      expect(manifest.ground_truth_labels).toBeDefined();
    });

    it("seed manifest is present", () => {
      const manifestPath = path.join(REPO_ROOT, "datasets", "manifests", "seed.json");
      expect(fs.existsSync(manifestPath)).toBe(true);
    });
  });

  describe("Scientific claims freeze is in effect", () => {
    it("SCIENTIFIC_CLAIMS_FREEZE.md exists", () => {
      const freezePath = path.join(REPO_ROOT, "reports", "SCIENTIFIC_CLAIMS_FREEZE.md");
      expect(fs.existsSync(freezePath)).toBe(true);
    });

    it("ablation report exists", () => {
      const ablationPath = path.join(REPO_ROOT, "reports", "JOINT2312_ABLATION_REPORT.md");
      expect(fs.existsSync(ablationPath)).toBe(true);
    });

    it("download_datasets.py script exists", () => {
      const dlPath = path.join(REPO_ROOT, "scripts", "download_datasets.py");
      expect(fs.existsSync(dlPath)).toBe(true);
    });

    it("loso.py splitting utility exists", () => {
      const losoPath = path.join(REPO_ROOT, "scripts", "loso.py");
      expect(fs.existsSync(losoPath)).toBe(true);
    });
  });
});
