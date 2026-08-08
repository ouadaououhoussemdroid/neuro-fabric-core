/**
 * T-016 — EEGPT adapter test (real ONNX-backed).
 *
 * The original stub test asserted implemented:false and NotImplementedError.
 * After checkpoint verification and ONNX export, the adapter now delegates to
 * ONNXAdapter with a real artifact at /models/eegpt-encoder-int8.onnx.
 */
import { describe, it, expect } from "vitest";
import { EEGPTAdapter } from "../eegpt-adapter";
import { type OrtRuntime } from "../onnx-adapter";
import { NotImplementedError, type ModelInput } from "../../types";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";

/**
 * Node-compatible runtime (real onnxruntime-web, CPU EP).
 *
 * Clears any `wasmPaths` override left by a prior `defaultRuntime()` call in
 * this file (the "load() throws in Node env" test exercises defaultRuntime,
 * which pins `wasmPaths = "/ort/"` on the shared onnxruntime-web singleton).
 * ORT must resolve its WASM bundle from node_modules — the same default that
 * makes Gate 5's nodeRuntime work — so we restore `wasmPaths` to undefined.
 */
async function nodeRuntime(): Promise<OrtRuntime> {
  const mod = (await import("onnxruntime-web")) as unknown as OrtRuntime & {
    env?: { wasm?: { wasmPaths?: string | Record<string, string> } };
  };
  if (mod?.env?.wasm) {
    mod.env.wasm.wasmPaths = undefined;
  }
  return mod as OrtRuntime;
}

/** 62-ch × 1000 @ 250 Hz input matching the EEGPT descriptor contract. */
function makeEEGPTInput(): ModelInput {
  const channels = 62;
  const samples = 1000;
  const sr = 250;
  const data = Array.from({ length: channels }, (_, c) =>
    Array.from({ length: samples }, (_, t) => Math.sin((2 * Math.PI * (c + 1) * t) / sr) * 0.5),
  );
  return { kind: "windows", windows: [{ data, sampleRate: sr, start: 0, end: samples }] };
}

const ARTEFACT_PATH = join(process.cwd(), "public", "models", "eegpt-encoder-int8.onnx");

function loadManifest() {
  return JSON.parse(
    readFileSync(join(process.cwd(), "public", "models", "manifest.json"), "utf-8"),
  );
}

describe("T-016 EEGPT adapter (real ONNX-backed)", () => {
  // Fresh adapter for each test to avoid state leakage from load() calls.
  function freshAdapter() {
    return new EEGPTAdapter();
  }

  it("declares implemented: true in its descriptor", () => {
    expect(freshAdapter().descriptor.capabilities.implemented).toBe(true);
  });

  it("marks itself as experimental (foundation model)", () => {
    expect(freshAdapter().descriptor.isExperimental).toBe(true);
  });

  it("uses kind 'eegpt'", () => {
    expect(freshAdapter().descriptor.kind).toBe("eegpt");
  });

  it("descriptor name reflects real implementation", () => {
    const name = freshAdapter().descriptor.name;
    expect(name).toContain("EEGPT");
    expect(name).not.toContain("Scheduled");
    expect(name).not.toContain("Planned");
  });

  it("descriptor references the real checkpoint and license", () => {
    const desc = freshAdapter().descriptor.description;
    expect(desc).toContain("braindecode/eegpt-pretrained");
    expect(desc).toContain("Apache-2.0");
    expect(desc).toContain("INT8");
  });

  it("artifactUri points to the deployed ONNX file", () => {
    expect(freshAdapter().descriptor.artifactUri).toBe("/models/eegpt-encoder-int8.onnx");
  });

  it("descriptor reports correct capabilities (62-ch, 250Hz, 1000-sample window)", () => {
    const caps = freshAdapter().descriptor.capabilities;
    expect(caps.channels).toBe(62);
    expect(caps.sampleRate).toBe(250);
    expect(caps.windowSamples).toBe(1000);
    expect(caps.embeddingDim).toBe(2048);
    // The [1,31,2048] ONNX output is mean-pooled over the 31 token axis
    // (NOT flattened to 63,488) — the descriptor must declare this contract.
    expect(caps.outputPooling).toBe("mean-tokens");
  });

  it("descriptor marks wasmCompatible: true", () => {
    expect(freshAdapter().descriptor.capabilities.wasmCompatible).toBe(true);
  });

  it("the ONNX artefact exists at the expected path", () => {
    const buf = readFileSync(ARTEFACT_PATH);
    expect(buf.length).toBeGreaterThan(1_000_000);
    // ONNX magic bytes
    expect(buf[0]).toBe(0x08);
  });

  it("manifest SHA-256 matches the deployed artefact", () => {
    const buf = readFileSync(ARTEFACT_PATH);
    const hash = createHash("sha256").update(buf).digest("hex");
    const manifest = loadManifest();
    // Manifest uses filename-based keys, not registry IDs
    const entry = manifest.models["eegpt-encoder-int8"];
    expect(entry).toBeDefined();
    expect(entry.sha256).toBe(hash);
    expect(entry.size).toBe(buf.length);
    expect(entry.wasmCompatible).toBe(true);
  });

  it("manifest registryId maps to the EEGPT adapter", () => {
    const manifest = loadManifest();
    const entry = manifest.models["eegpt-encoder-int8"];
    expect(entry).toBeDefined();
    expect(entry.registryId).toBe("onnx-eegpt");
  });

  it("isLoaded() returns false on a fresh adapter", () => {
    expect(freshAdapter().isLoaded()).toBe(false);
  });

  it("unload() is safe before load()", async () => {
    await expect(freshAdapter().unload()).resolves.toBeUndefined();
  });

  // VERIFICATION — real ONNX inference through EEGPTAdapter → ONNXAdapter →
  // onnxruntime-web (Node CPU EP). Proves the [1,31,2048] output is mean-pooled
  // to exactly 2048 and that no silent fallback/zero-vector is returned.
  //
  // Must run BEFORE the "load() throws in Node" test below: that test exercises
  // `defaultRuntime()`, which pins `wasmPaths = "/ort/"` on the shared
  // onnxruntime-web singleton and caches a failed WASM backend init, which would
  // make this real-inference test reuse the broken backend. Running first gives
  // the WASM backend a clean init (the same state Gate 5 relies on).
  it("runs real ONNX inference through EEGPTAdapter and returns exactly 2048-dim", async () => {
    const adapter = new EEGPTAdapter({
      artifact: ARTEFACT_PATH,
      runtime: nodeRuntime,
      executionProviders: ["wasm"],
    });
    await adapter.load();
    expect(adapter.isLoaded()).toBe(true);

    const out = await adapter.embed(makeEEGPTInput());
    expect(out.modelId).toBe("onnx-eegpt");
    // Exact 2048-d contract (NOT the 63,488-dim flatten).
    expect(out.dim).toBe(2048);
    expect(out.vector).toHaveLength(2048);

    // Non-degenerate output: proves real inference, not a silent zero/NaN fallback.
    const sum = out.vector.reduce((a, b) => a + Math.abs(b), 0);
    expect(sum).toBeGreaterThan(0);
    for (const v of out.vector) expect(Number.isFinite(v)).toBe(true);

    await adapter.unload();
    expect(adapter.isLoaded()).toBe(false);
  }, 60000);

  it("load() throws when onnxruntime-web is unavailable (Node env)", async () => {
    const adapter = freshAdapter();
    // In the Node test environment, onnxruntime-web's InferenceSession.create
    // will fail because there's no WASM backend. The adapter should propagate
    // the error (not silently succeed or return NotImplementedError).
    await expect(adapter.load()).rejects.toThrow();
  });

  it("embed() throws NotImplementedError on a fresh adapter (before load())", async () => {
    const adapter = freshAdapter(); // fresh, not yet loaded
    await expect(
      adapter.embed({
        kind: "windows",
        windows: [
          {
            data: Array.from({ length: 62 }, () => new Array(1000).fill(0)),
            sampleRate: 250,
            start: 0,
            end: 1000,
          },
        ],
      }),
    ).rejects.toThrow(NotImplementedError);
  });

  it("predict() throws NotImplementedError on a fresh adapter (before load())", async () => {
    const adapter = freshAdapter();
    await expect(
      adapter.predict({
        kind: "windows",
        windows: [
          {
            data: Array.from({ length: 62 }, () => new Array(1000).fill(0)),
            sampleRate: 250,
            start: 0,
            end: 1000,
          },
        ],
      }),
    ).rejects.toThrow(NotImplementedError);
  });
});
