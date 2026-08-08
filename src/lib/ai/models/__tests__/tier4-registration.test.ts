/**
 * T-016 / Tier 4 — Registry integration tests for real ONNX artefacts.
 *
 * Verifies that every Tier 4 foundation model is registered in the platform
 * registry (src/lib/ai/models/registry.ts) with correct metadata, pointing to
 * a real deployed ONNX artifact in public/models/, and that the full
 * registry → adapter → ONNX path is sound.
 */
import { describe, it, expect } from "vitest";
import { hasModel, listModels, getDescriptor, createAdapter } from "../registry";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";

const MODELS_DIR = join(process.cwd(), "public", "models");
const MANIFEST = JSON.parse(readFileSync(join(MODELS_DIR, "manifest.json"), "utf-8"));

/** Manifest key → filename. Manifest uses filenames-without-ext as keys. */
const MANIFEST_KEY_FOR: Record<string, string> = {
  "braindecode-eegconformer-prod": "eegconformer",
  "onnx-eegpt": "eegpt-encoder-int8",
  "onnx-femba-tiny": "femba-tiny-encoder-adapter",
  "onnx-labram": "labram-encoder",
  "onnx-cbramod": "cbramod-encoder",
};

/** Read the bytes of the artifact that backs a given registry model id. */
function artefactBytes(registryId: string): Buffer {
  const manifestKey = MANIFEST_KEY_FOR[registryId];
  expect(manifestKey, `no manifest key mapping for "${registryId}"`).toBeDefined();
  const entry = MANIFEST.models[manifestKey];
  expect(entry, `manifest missing entry for "${manifestKey}"`).toBeDefined();
  const p = join(process.cwd(), "public", entry.url);
  return readFileSync(p);
}

function verifyArtifact(
  registryId: string,
  expectedWasm: boolean,
  expectedBlockers?: string[],
): void {
  const manifestKey = MANIFEST_KEY_FOR[registryId];
  const entry = MANIFEST.models[manifestKey];
  expect(entry).toBeDefined();
  expect(entry.registryId).toBe(registryId);

  const buf = readFileSync(join(process.cwd(), "public", entry.url));
  const hash = createHash("sha256").update(buf).digest("hex");
  expect(entry.sha256).toBe(hash);
  expect(entry.size).toBe(buf.length);
  // ONNX magic byte
  expect(buf[0]).toBe(0x08);
  // WASM compatibility
  expect(entry.wasmCompatible).toBe(expectedWasm);
  if (expectedBlockers) {
    expect(entry.wasmBlockers).toEqual(expect.arrayContaining(expectedBlockers));
  }
}

describe("Tier 4 model registry integration", () => {
  // ── Model presence ──────────────────────────────────────────────
  describe("all 5 Tier 4 models are registered", () => {
    const expected = ["onnx-eegpt", "onnx-femba-tiny", "onnx-labram", "onnx-cbramod"];
    const alsoExpected = ["braindecode-eegconformer-prod"];

    it.each([...expected, ...alsoExpected])("%s is registered", (id) => {
      expect(hasModel(id), `${id} not in registry`).toBe(true);
    });

    it("listModels() includes all Tier 4 ids", () => {
      const ids = listModels().map((m) => m.id);
      expect(ids).toContain("braindecode-eegconformer-prod");
      expect(ids).toContain("onnx-eegpt");
      expect(ids).toContain("onnx-femba-tiny");
      expect(ids).toContain("onnx-labram");
      expect(ids).toContain("onnx-cbramod");
    });
  });

  // ── No duplicate EEGNetv4 ───────────────────────────────────────
  it("EEGNetv4 is registered exactly once", () => {
    const ids = listModels().map((m) => m.id);
    expect(ids.filter((id) => id === "braindecode-eegnetv4-default")).toHaveLength(1);
  });

  // ── Per-model descriptor + artefact checks ──────────────────────
  describe("EEGPT (onnx-eegpt)", () => {
    const d = getDescriptor("onnx-eegpt")!;

    it("descriptor has correct metadata", () => {
      expect(d.kind).toBe("eegpt");
      expect(d.capabilities.channels).toBe(62);
      expect(d.capabilities.sampleRate).toBe(250);
      expect(d.capabilities.windowSamples).toBe(1000);
      expect(d.capabilities.runtime).toBe("wasm");
      expect(d.capabilities.implemented).toBe(true);
      expect(d.capabilities.wasmCompatible).toBe(true);
    });

    it("artifactUri points to the deployed ONNX file", () => {
      expect(d.artifactUri).toBe("/models/eegpt-encoder-int8.onnx");
    });

    it("artifact file exists, SHA-256 matches, and is WASM-compatible", () => {
      verifyArtifact("onnx-eegpt", true);
    });

    it("createAdapter() returns an adapter with embed()", () => {
      const adapter = createAdapter("onnx-eegpt");
      expect(adapter.embed).toBeDefined();
      expect(typeof adapter.embed).toBe("function");
    });
  });

  describe("FEMBA-tiny (onnx-femba-tiny)", () => {
    const d = getDescriptor("onnx-femba-tiny")!;

    it("descriptor has correct metadata", () => {
      expect(d.kind).toBe("onnx");
      expect(d.capabilities.channels).toBe(22);
      expect(d.capabilities.sampleRate).toBe(200);
      expect(d.capabilities.windowSamples).toBe(1280);
      expect(d.capabilities.implemented).toBe(true);
      expect(d.capabilities.wasmCompatible).toBe(true);
    });

    it("artifactUri points to the adapter ONNX", () => {
      expect(d.artifactUri).toBe("/models/femba-tiny-encoder-adapter.onnx");
    });

    it("artifact file exists, SHA-256 matches, and is WASM-compatible", () => {
      verifyArtifact("onnx-femba-tiny", true);
    });

    it("createAdapter() returns an adapter with embed()", () => {
      const adapter = createAdapter("onnx-femba-tiny");
      expect(adapter.embed).toBeDefined();
    });
  });

  describe("LaBraM (onnx-labram)", () => {
    const d = getDescriptor("onnx-labram")!;

    it("descriptor has correct metadata", () => {
      expect(d.kind).toBe("onnx");
      expect(d.capabilities.channels).toBe(16);
      expect(d.capabilities.sampleRate).toBe(250);
      expect(d.capabilities.windowSamples).toBe(1600);
      expect(d.capabilities.implemented).toBe(true);
      expect(d.capabilities.wasmCompatible).toBe(true);
    });

    it("artifactUri is set to the deployed ONNX", () => {
      expect(d.artifactUri).toBe("/models/labram-encoder.onnx");
    });

    it("artifact file exists, SHA-256 matches, and is WASM-compatible", () => {
      verifyArtifact("onnx-labram", true);
    });

    it("createAdapter() returns an adapter with embed()", () => {
      const adapter = createAdapter("onnx-labram");
      expect(adapter.embed).toBeDefined();
    });
  });

  describe("CBraMod (onnx-cbramod)", () => {
    const d = getDescriptor("onnx-cbramod")!;

    it("descriptor has correct metadata", () => {
      expect(d.kind).toBe("onnx");
      expect(d.capabilities.channels).toBe(19);
      expect(d.capabilities.sampleRate).toBe(250);
      expect(d.capabilities.windowSamples).toBe(1000);
      expect(d.capabilities.implemented).toBe(true);
    });

    it("is marked as NOT WASM-compatible", () => {
      expect(d.capabilities.wasmCompatible).toBe(false);
    });

    it("has wasmBlockers listing DFT and ReduceL2", () => {
      expect(d.capabilities.wasmBlockers).toContain("DFT");
      expect(d.capabilities.wasmBlockers).toContain("ReduceL2");
    });

    it("artifact file exists, SHA-256 matches, and is marked WASM-incompatible", () => {
      verifyArtifact("onnx-cbramod", false, ["DFT", "ReduceL2"]);
    });

    it("createAdapter() returns an adapter with embed()", () => {
      const adapter = createAdapter("onnx-cbramod");
      expect(adapter.embed).toBeDefined();
    });
  });

  // ── Manifest ↔ registry cross-check ─────────────────────────────
  describe("manifest ↔ registry consistency", () => {
    it("every Tier 4 registry artifact has a manifest entry with matching SHA-256", () => {
      for (const [regId, manifestKey] of Object.entries(MANIFEST_KEY_FOR)) {
        const d = getDescriptor(regId);
        expect(d, `${regId} not registered`).toBeDefined();
        const entry = MANIFEST.models[manifestKey];
        expect(entry, `${manifestKey} not in manifest`).toBeDefined();
        const buf = readFileSync(join(process.cwd(), "public", entry.url));
        const hash = createHash("sha256").update(buf).digest("hex");
        expect(entry.sha256, `${manifestKey} SHA-256 mismatch`).toBe(hash);
      }
    });

    it("manifest wasmCompatible flags match registry descriptors", () => {
      const expected = {
        "onnx-eegpt": true,
        "onnx-femba-tiny": true,
        "onnx-labram": true,
        "onnx-cbramod": false,
        "braindecode-eegconformer-prod": true,
      };
      for (const [regId, compat] of Object.entries(expected)) {
        const d = getDescriptor(regId);
        expect(d, `${regId} not registered`).toBeDefined();
        expect(d!.capabilities.wasmCompatible, `${regId} wasmCompatible mismatch`).toBe(compat);
      }
    });
  });

  // ── Registry count ──────────────────────────────────────────────
  it("registry contains all expected models", () => {
    const ids = listModels().map((m) => m.id);
    // Tier 4 models
    for (const t of [
      "braindecode-eegconformer-prod",
      "onnx-eegpt",
      "onnx-femba-tiny",
      "onnx-labram",
      "onnx-cbramod",
    ]) {
      expect(ids).toContain(t);
    }
    // PCA baseline must still exist
    expect(ids).toContain("pca-legacy-v1");
  });
});
