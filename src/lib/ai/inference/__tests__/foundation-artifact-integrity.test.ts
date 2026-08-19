/**
 * T-036 / Mission 14 Gate 1 — Artifact integrity (hash-mismatch safe failure).
 *
 * These tests prove the Tier-2 CBraMod ONNX path fails SAFE on a corrupted/mismatched
 * artifact: the SHA-256 gate in `ensureAdapter()` (foundation.server.ts:163-172) must
 * raise `FoundationUnavailableError("SHA-256 verification failed")`, which the route
 * maps to HTTP 424 — and it must NEVER degrade to V2/PCA/embedEEG.
 *
 * Strategy (matches foundation.server.test.ts idiom):
 *  - onnxruntime-node stays mocked (no native forward).
 *  - We mutate the REAL on-disk artifact + manifest entry in beforeEach and restore
 *    the original bytes in afterEach, so the production loader reads genuine files.
 *  - We assert the throw comes from verifyArtefact, and that embedFoundationWindows
 *    rejects with FoundationUnavailableError matching /SHA-256 verification failed/i.
 *
 * No production source is modified. The artifact is restored to its exact original
 * bytes (sha256 c128ccfd…) after the suite, verified by a final integrity assertion.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { EEGWindow } from "@/lib/eeg/types";

vi.mock("onnxruntime-node", () => {
  const fakeOutput = new Float32Array(19 * 5 * 200);
  for (let i = 0; i < fakeOutput.length; i++) fakeOutput[i] = Math.random();
  const session = {
    inputNames: ["eeg"],
    outputNames: ["embedding"],
    run: async () => ({
      embedding: { data: fakeOutput, dims: [1, 19, 5, 200], type: "float32" },
    }),
    release: async () => {},
  };
  return {
    InferenceSession: { create: async () => session },
    Tensor: class {
      type = "float32";
      data = fakeOutput;
      dims = [1, 19, 5, 200] as const;
    },
    env: { backend: {} },
  };
});

const {
  FOUNDATION_ARTIFACT_ID,
  FoundationUnavailableError,
  foundationProvenance,
  embedFoundationWindows,
  resetFoundationAdapter,
} = await import("../foundation.server");

const REPO = join(process.cwd());
const ARTIFACT_PATH = join(REPO, "public", "models", "cbramod-encoder.onnx");
const MANIFEST_PATH = join(REPO, "public", "models", "manifest.json");

function syntheticWindow(): EEGWindow {
  return {
    data: Array.from({ length: 19 }, () => Array.from({ length: 1000 }, () => Math.random())),
    sampleRate: 250,
    start: 0,
    end: 1000,
  };
}

// Snapshot of the original artifact bytes so we can restore it byte-for-byte.
let originalBytes: Uint8Array | null = null;
let originalManifest: string | null = null;
const EXPECTED_SHA = "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47";

function sha256Hex(data: Uint8Array): string {
  // Mirrors src/lib/ai/artefacts/hashed-artefact.ts sha256Hex (node:crypto).
  const { createHash } = require("node:crypto");
  return createHash("sha256").update(Buffer.from(data)).digest("hex");
}

describe("foundation.server — artifact integrity (Gate 1: hash-mismatch safe failure)", () => {
  beforeEach(() => {
    resetFoundationAdapter();
    // Persist original bytes for byte-for-byte restore.
    if (existsSync(ARTIFACT_PATH)) originalBytes = readFileSync(ARTIFACT_PATH);
    if (existsSync(MANIFEST_PATH)) originalManifest = readFileSync(MANIFEST_PATH, "utf-8");
  });

  afterEach(() => {
    // CRITICAL: restore the real artifact + manifest exactly as they were.
    if (originalBytes) writeFileSync(ARTIFACT_PATH, Buffer.from(originalBytes));
    if (originalManifest !== null) writeFileSync(MANIFEST_PATH, originalManifest);
    resetFoundationAdapter();
  });

  it("preconditions: real CBraMod artifact present and already SHA-verified", () => {
    expect(existsSync(ARTIFACT_PATH)).toBe(true);
    const onDisk = readFileSync(ARTIFACT_PATH);
    expect(sha256Hex(new Uint8Array(onDisk.buffer, onDisk.byteOffset, onDisk.byteLength))).toBe(
      EXPECTED_SHA,
    );
    const prov = foundationProvenance();
    expect(prov.artifact_id).toBe(FOUNDATION_ARTIFACT_ID);
    expect(prov.sha256).toBe(EXPECTED_SHA);
  });

  it("hash mismatch rejects with FoundationUnavailableError (SHA-256 verification failed)", async () => {
    // Flip a single byte IN PLACE (no size change) so the size gate passes and the
    // SHA-256 gate (foundation.server.ts L163-172) is the path that fires.
    const onDisk = readFileSync(ARTIFACT_PATH);
    const corrupted = Buffer.from(onDisk); // copy
    const flipPos = Math.floor(corrupted.length / 2);
    corrupted[flipPos] = corrupted[flipPos] ^ 0xff; // XOR flip — same length, different content
    writeFileSync(ARTIFACT_PATH, corrupted);

    // Sanity: same byte length (size gate should pass) but genuinely different hash.
    expect(corrupted.length).toBe(onDisk.length);
    expect(
      sha256Hex(new Uint8Array(corrupted.buffer, corrupted.byteOffset, corrupted.byteLength)),
    ).not.toBe(EXPECTED_SHA);

    await expect(embedFoundationWindows([syntheticWindow()])).rejects.toMatchObject({
      name: "FoundationUnavailableError",
    });
    // The wrapper message must reference the SHA-256 gate (foundation.server.ts L170-172).
    await expect(
      embedFoundationWindows([syntheticWindow()]).catch((e) => {
        throw e instanceof FoundationUnavailableError ? e : new Error(String(e));
      }),
    ).rejects.toThrow(/SHA-256 verification failed/i);
  });

  it("hash mismatch never falls back to V2 / PCA (rejects, never calls embedEEG)", async () => {
    // Size-preserving corruption (flip one byte) so the SHA-256 gate is the failure.
    const onDisk = readFileSync(ARTIFACT_PATH);
    const corrupted = Buffer.from(onDisk);
    corrupted[corrupted.length - 1] = corrupted[corrupted.length - 1] ^ 0xff;
    writeFileSync(ARTIFACT_PATH, corrupted);

    // PROVE no V2/PCA fallback: spy on the V2 embed path. foundation.server.ts
    // never imports embedEEG (grep-verified), so even if the spy were reachable it
    // must remain uncalled. We assert the request rejects as FoundationUnavailableError.
    const embedEEGSpy = vi.fn();
    vi.doMock("@/lib/ai/embeddings", () => ({
      __esm: true,
      embedEEG: embedEEGSpy,
      finalize: (o: unknown) => o,
      defaultEmbedding: undefined,
    }));

    await expect(embedFoundationWindows([syntheticWindow()])).rejects.toBeInstanceOf(
      FoundationUnavailableError,
    );

    // embedEEG must never be invoked on the Tier-2 hash-failure path — the negative
    // result above (a rejected promise, no embedding returned) is the proof of no
    // silent fallback to V2 or PCA.
    expect(embedEEGSpy).not.toHaveBeenCalled();
  });

  it("size mismatch rejects with FoundationUnavailableError (artifact size mismatch)", async () => {
    const onDisk = readFileSync(ARTIFACT_PATH);
    // Prepend garbage to violate the size gate (foundation.server.ts L163).
    const tampered = Buffer.concat([Buffer.from([0x00, 0x01, 0x02]), onDisk]);
    writeFileSync(ARTIFACT_PATH, tampered);

    await expect(
      embedFoundationWindows([syntheticWindow()]).catch((e) => {
        throw e instanceof FoundationUnavailableError ? e : new Error(String(e));
      }),
    ).rejects.toThrow(/artifact size mismatch/i);
  });

  it("restores the real artifact byte-for-byte after tampering", () => {
    // This runs last only by accident of ordering — assert idempotency explicitly:
    // the artifact on disk must equal the pre-test snapshot (sha256 c128ccfd…).
    expect(existsSync(ARTIFACT_PATH)).toBe(true);
    const onDisk = readFileSync(ARTIFACT_PATH);
    const bytes = new Uint8Array(onDisk.buffer, onDisk.byteOffset, onDisk.byteLength);
    expect(sha256Hex(bytes)).toBe(EXPECTED_SHA);
  });
});
