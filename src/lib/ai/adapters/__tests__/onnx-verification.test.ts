/**
 * T-016 — Runtime SHA-256 artifact verification tests.
 *
 * Verifies that when `enableVerification` is set on an ONNXAdapter:
 *   - Valid artifacts pass verification and load() succeeds
 *   - Tampered artifacts (hash mismatch) cause load() to throw
 *   - Size mismatches cause load() to throw
 *   - Missing manifest entries skip verification (backward compatible)
 *   - Fetch failures cause load() to throw
 *   - Without enableVerification, no fetch is attempted (backward compatible)
 *
 * Also verifies that a verification failure in the embed() facade
 * triggers the PCA fallback chain.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  ONNXAdapter,
  type OrtRuntime,
  type OrtSessionLike,
  type OrtTensorLike,
} from "../onnx-adapter";
import { sha256Hex } from "../../artefacts/hashed-artefact";
import { __resetManifestCache } from "../../artefacts/runtime-verifier";
import { metrics, resetMetrics } from "../../../metrics";
import { embed } from "../../embeddings/index";
import { registerModel, unregisterModel } from "../../models/registry";

const FAKE_ARTIFACT_URL = "/models/test-model.onnx";
const FAKE_ARTIFACT_BYTES = new Uint8Array([0x08, 0x06, 0x0a, 0x02, 0x18, 0x02]);

function makeManifestEntry(url: string, sha256: string, size: number) {
  return { url, sha256, size };
}

function makeManifest(entries: Record<string, ReturnType<typeof makeManifestEntry>>) {
  return {
    generated: "2026-08-09T00:00:00.000Z",
    models: entries,
  };
}

function makeFakeRuntime(outputDim = 4): OrtRuntime {
  const session: OrtSessionLike = {
    inputNames: ["input"],
    outputNames: ["embedding"],
    async run(feeds) {
      const t = feeds["input"];
      const sum = Array.from(t.data as ArrayLike<number>).reduce((a, b) => a + Number(b), 0);
      const data = Float32Array.from({ length: outputDim }, (_, i) => sum + i);
      return { embedding: { data, dims: [1, outputDim] } satisfies OrtTensorLike };
    },
    async release() {},
  };
  return {
    InferenceSession: {
      async create() {
        return session;
      },
    },
    Tensor: class {
      constructor(
        public type: "float32",
        public data: Float32Array,
        public dims: readonly number[],
      ) {}
    } as unknown as OrtRuntime["Tensor"],
  };
}

/**
 * Stub globalThis.fetch to return a manifest JSON for any URL containing
 * "manifest.json", and return the provided bytes for the artifact URL.
 */
function stubFetch(
  manifestContent: unknown,
  artifactBytes?: Uint8Array,
  options: { artifactStatus?: number } = {},
) {
  const manifestJson = JSON.stringify(manifestContent);

  const fakeFetch = vi.fn(async (url: string | Request) => {
    const urlStr = typeof url === "string" ? url : url.toString();

    if (urlStr.includes("manifest.json")) {
      return {
        ok: true,
        json: async () => manifestContent,
        arrayBuffer: async () => new TextEncoder().encode(manifestJson).buffer,
      } as unknown as Response;
    }

    if (urlStr === FAKE_ARTIFACT_URL) {
      if (artifactBytes) {
        return {
          ok: true,
          arrayBuffer: async () => artifactBytes.buffer,
        } as unknown as Response;
      }
      return { ok: false, status: options.artifactStatus ?? 404 } as unknown as Response;
    }

    // External data file
    if (urlStr.endsWith(".onnx.data")) {
      return {
        ok: true,
        arrayBuffer: async () => new Uint8Array([0, 0, 0]).buffer,
      } as unknown as Response;
    }

    return { ok: false, status: 404 } as unknown as Response;
  });

  vi.stubGlobal("fetch", fakeFetch);
  return { fakeFetch };
}

function makeVerifiedAdapter(opts: { enableVerification: boolean }) {
  return new ONNXAdapter({
    id: "test-verified",
    name: "Test",
    version: "0.0.1",
    description: "fake",
    artifact: FAKE_ARTIFACT_URL,
    task: "embedding",
    inputShape: { kind: "features", dim: 3 },
    runtime: async () => makeFakeRuntime(8),
    enableVerification: opts.enableVerification,
  });
}

function makeWindowInput() {
  const C = 2;
  const T = 256;
  const sampleRate = 128;
  const data: number[][] = [];
  for (let c = 0; c < C; c++) {
    const ch = new Array<number>(T);
    for (let t = 0; t < T; t++) ch[t] = Math.sin((2 * Math.PI * (8 + c) * t) / sampleRate);
    data.push(ch);
  }
  return {
    kind: "windows" as const,
    windows: [{ data, sampleRate, start: 0, end: T }],
  };
}

describe("ONNXAdapter runtime SHA-256 verification (T-016)", () => {
  beforeEach(() => {
    resetMetrics();
    __resetManifestCache();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("load() with enableVerification + valid artifact succeeds", async () => {
    const bytes = new Uint8Array(FAKE_ARTIFACT_BYTES);
    const correctHash = sha256Hex(bytes);
    const manifest = makeManifest({
      "test-model": makeManifestEntry(FAKE_ARTIFACT_URL, correctHash, bytes.length),
    });

    stubFetch(manifest, bytes);

    const adapter = makeVerifiedAdapter({ enableVerification: true });
    await adapter.load();
    expect(adapter.isLoaded()).toBe(true);

    // Artifact verification metrics should reflect the pass.
    expect(metrics.artifactVerificationTotal.value({ result: "attempt" })).toBe(1);
    expect(metrics.artifactVerificationTotal.value({ result: "pass" })).toBe(1);
    expect(metrics.artifactVerificationTotal.value({ result: "fail" })).toBe(0);

    await adapter.unload();
  });

  it("load() throws on SHA-256 hash mismatch (tampered artifact)", async () => {
    const bytes = new Uint8Array(FAKE_ARTIFACT_BYTES);
    // Deliberately wrong hash — will not match the actual bytes.
    const manifest = makeManifest({
      "test-model": makeManifestEntry(FAKE_ARTIFACT_URL, "d".repeat(64), bytes.length),
    });

    stubFetch(manifest, bytes);

    const adapter = makeVerifiedAdapter({ enableVerification: true });
    await expect(adapter.load()).rejects.toThrow(/artifact verification failed/i);
    expect(adapter.isLoaded()).toBe(false);

    // Verification metrics should reflect the fail (hash mismatch).
    expect(metrics.artifactVerificationTotal.value({ result: "attempt" })).toBe(1);
    expect(metrics.artifactVerificationTotal.value({ result: "pass" })).toBe(0);
    expect(
      metrics.artifactVerificationTotal.value({ result: "fail", reason: "hash_mismatch" }),
    ).toBe(1);
  });

  it("load() throws on size mismatch", async () => {
    const bytes = new Uint8Array(FAKE_ARTIFACT_BYTES);
    const correctHash = sha256Hex(bytes);
    const manifest = makeManifest({
      // Correct hash but wrong size — size check runs first.
      "test-model": makeManifestEntry(FAKE_ARTIFACT_URL, correctHash, 999999),
    });

    stubFetch(manifest, bytes);

    const adapter = makeVerifiedAdapter({ enableVerification: true });
    await expect(adapter.load()).rejects.toThrow(/size mismatch/i);

    // Verification metrics should reflect the fail (size mismatch).
    expect(metrics.artifactVerificationTotal.value({ result: "attempt" })).toBe(1);
    expect(
      metrics.artifactVerificationTotal.value({ result: "fail", reason: "size_mismatch" }),
    ).toBe(1);
  });

  it("load() throws on fetch failure (404 for artifact)", async () => {
    const bytes = new Uint8Array(FAKE_ARTIFACT_BYTES);
    const correctHash = sha256Hex(bytes);
    const manifest = makeManifest({
      "test-model": makeManifestEntry(FAKE_ARTIFACT_URL, correctHash, bytes.length),
    });

    // Return manifest but fail the artifact fetch.
    stubFetch(manifest, undefined, { artifactStatus: 404 });

    const adapter = makeVerifiedAdapter({ enableVerification: true });
    await expect(adapter.load()).rejects.toThrow(/fetch failed/i);

    // Verification metrics should reflect the fail (fetch error).
    expect(metrics.artifactVerificationTotal.value({ result: "attempt" })).toBe(1);
    expect(metrics.artifactVerificationTotal.value({ result: "fail", reason: "fetch_error" })).toBe(
      1,
    );
  });

  it("load() skips verification when no manifest entry matches the URL (backward compatible)", async () => {
    const bytes = new Uint8Array(FAKE_ARTIFACT_BYTES);
    // Manifest has an entry, but for a different URL.
    const manifest = makeManifest({
      "other-model": makeManifestEntry("/models/other-model.onnx", "0".repeat(64), 100),
    });

    stubFetch(manifest, bytes);

    const adapter = makeVerifiedAdapter({ enableVerification: true });
    // resolveVerification returns null — verification skipped, load succeeds.
    await adapter.load();
    expect(adapter.isLoaded()).toBe(true);

    await adapter.unload();
  });

  it("load() without enableVerification does not call fetch (backward compatible)", async () => {
    const bytes = new Uint8Array(FAKE_ARTIFACT_BYTES);
    const manifest = makeManifest({
      "test-model": makeManifestEntry(FAKE_ARTIFACT_URL, sha256Hex(bytes), bytes.length),
    });

    const { fakeFetch } = stubFetch(manifest, bytes);

    const adapter = new ONNXAdapter({
      id: "test-no-verify",
      name: "Test",
      version: "0.0.1",
      description: "",
      artifact: FAKE_ARTIFACT_URL,
      task: "embedding",
      inputShape: { kind: "features", dim: 3 },
      runtime: async () => makeFakeRuntime(8),
      // enableVerification NOT set — backward compatible.
    });

    await adapter.load();
    expect(adapter.isLoaded()).toBe(true);
    // Verification should NOT have been attempted — no fetch calls at all.
    expect(fakeFetch).not.toHaveBeenCalled();

    await adapter.unload();
  });
});

describe("Verification failure triggers PCA fallback (T-016)", () => {
  beforeEach(() => {
    resetMetrics();
    __resetManifestCache();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    unregisterModel("test-verify-fallback");
  });

  it("embed() falls back to PCA when artifact hash verification fails", async () => {
    const bytes = new Uint8Array(FAKE_ARTIFACT_BYTES);
    // Manifest has wrong hash → verification fails → PCA fallback.
    const manifest = makeManifest({
      "test-model": makeManifestEntry(FAKE_ARTIFACT_URL, "deadbeef".repeat(8), bytes.length),
    });

    stubFetch(manifest, bytes);

    // Register a model with verification enabled. Its load() will fail the
    // hash check, so embed() should fall back to PCA.
    registerModel(
      () =>
        new ONNXAdapter({
          id: "test-verify-fallback",
          name: "Test Verify Fallback",
          version: "0.0.1",
          description: "",
          artifact: FAKE_ARTIFACT_URL,
          enableVerification: true,
          task: "embedding",
          inputShape: { kind: "raw", channels: 2, samples: 256 },
          runtime: async () => makeFakeRuntime(4),
        }),
    );

    const result = await embed(makeWindowInput(), {
      modelId: "test-verify-fallback",
      fallbackToPCA: true,
    });

    // Should have fallen back to PCA.
    expect(result.fellBack).toBe(true);
    expect(result.modelId).toBe("pca-legacy-v1");
    expect(result.vector.length).toBeGreaterThan(0);
    expect(result.fallbackReason).toMatch(/artifact verification failed/);
  });
});
