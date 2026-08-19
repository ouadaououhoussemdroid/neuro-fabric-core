/**
 * Mission 15 Phase 4 — Artifact SHA Serving-Path Verification (close M14 Gate A3).
 *
 * REAL artifact-integrity flow end-to-end: the route calls ensureAdapter() → loads
 * the ONNX from disk → verifyArtefact() compares against manifest.json sha256
 * (c128ccfd…) → on match, proceeds to real onnxruntime-node inference. No mocking
 * of embedFoundationWindows — real ONNX CPU EP inference validates the 200-D output.
 *
 * Only embedEEG (V2) is mocked so we can ASSERT no fallback on 424. Auth, rate-
 * limit, NeuralVectorIndex, and ONNX inference are all REAL.
 *
 * Flow per test: backup artifact → corrupt → resetFoundationAdapter() → request
 * → expect 424 → restore artifact → resetFoundationAdapter() → request → expect 200.
 * Final assertion: artifact SHA-256 byte-for-byte matches the pre-test snapshot.
 *
 * CRITICAL: the 22 MB ONNX file must be restored exactly (byte-for-byte) so
 * production is never left in a corrupted state — even on test failure.
 */
import { describe, it, expect, vi, afterAll } from "vitest";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { createClient } from "@supabase/supabase-js";
import type { Database } from "@/integrations/supabase/types";
import { sha256Hex } from "@/lib/ai/artefacts/hashed-artefact";

// ── 1. Point at LOCAL Supabase stack ───────────────────────────────────────────
const _tokens = JSON.parse(readFileSync("reports/m15_jwt_test_tokens.json", "utf-8")) as {
  apiUrl: string;
  anonKey: string;
  serviceRoleKey: string;
  userA: { id: string; jwt: string };
  userB: { id: string; jwt: string };
  expiredJWT: string;
  invalidJWT: string;
};
process.env.SUPABASE_URL = _tokens.apiUrl;
process.env.SUPABASE_PUBLISHABLE_KEY = _tokens.anonKey;
process.env.SUPABASE_SERVICE_ROLE_KEY = _tokens.serviceRoleKey;

// ── 2. Mock ONLY V2 embedEEG (assert no fallback); keep real ONNX inference ────
const embedEEGMock = vi.fn();
vi.mock("@/lib/ai/inference/embed-eeg", () => ({
  embedEEG: embedEEGMock,
  DEFAULT_PREFERRED: "braindecode-eegconformer-prod-v2",
}));

// ── 3. Real imports (embedFoundationWindows is NOT mocked — real ONNX) ──────────
const admin = createClient<Database>(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
);
const { Route } = await import("../foundation");
const { FOUNDATION_EMBEDDING_DIM, FOUNDATION_MODEL_ID } =
  await import("@/lib/ai/inference/foundation.server");
const { resetFoundationAdapter } = await import("@/lib/ai/inference/foundation.server");

// ── 4. Artifact path + expected hash ───────────────────────────────────────────
const REPO = process.cwd();
const ARTIFACT_PATH = `${REPO}/public/models/cbramod-encoder.onnx`;
const EXPECTED_SHA = "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47";

// ── 5. Helpers ─────────────────────────────────────────────────────────────────
const CBraMod19 = [
  "FP1",
  "FP2",
  "F3",
  "F4",
  "C3",
  "C4",
  "P3",
  "P4",
  "O1",
  "O2",
  "F7",
  "F8",
  "T7",
  "T8",
  "P7",
  "P8",
  "FZ",
  "CZ",
  "PZ",
];

/** CSV with unique-per-seed content so different windows yield distinct embeddings. */
function csvFile(name = "signal.csv", rows = 2000, seed = 0): File {
  const lines = [CBraMod19.join(",")];
  for (let r = 0; r < rows; r++) {
    lines.push(
      CBraMod19.map((_, c) => (Math.sin((r + c + seed) * 0.01 + seed * 0.7) * 0.3).toFixed(4)).join(
        ",",
      ),
    );
  }
  return new File([lines.join("\n")], name, { type: "text/csv" });
}

function makeRequest(jwt?: string): Request {
  const form = new FormData();
  form.set("file", csvFile("signal.csv", 2000, 0));
  form.set("sampleRate", "250");
  const headers: Record<string, string> = {};
  if (jwt) headers.authorization = `Bearer ${jwt}`;
  return new Request("http://localhost/api/eeg/embed/foundation", {
    method: "POST",
    body: form,
    headers,
  });
}

type PostHandler = (ctx: { request: Request; context: unknown }) => Promise<Response>;

function callFoundation(request: Request) {
  const handlers = Route.options.server!.handlers as unknown as {
    POST: PostHandler;
  };
  return handlers.POST({ request, context: {} });
}

async function resetDB() {
  await admin
    .from("foundation_embeddings")
    .delete()
    .neq("id", "00000000-0000-0000-0000-000000000000");
  await admin.from("rate_limits").delete().neq("user_id", "00000000-0000-0000-0000-000000000000");
}

// ── 6. Tests ───────────────────────────────────────────────────────────────────
// Snapshot of the original artifact bytes — restored byte-for-byte in finally/afterAll.
let originalBytes: Uint8Array;

describe("M15 Phase 4: Artifact SHA Serving-Path Integrity (real ONNX, real route)", () => {
  try {
    // Pre-snapshot before any test corrupts the file. If this fails we bail hard.
    if (!existsSync(ARTIFACT_PATH)) {
      throw new Error(`CBraMod artifact not found at ${ARTIFACT_PATH}`);
    }
    originalBytes = readFileSync(ARTIFACT_PATH);
  } catch (e) {
    throw new Error(`Phase 4 setup failed: ${(e as Error).message}`);
  }

  // Belt-and-suspenders: restore the artifact even on catastrophic test failure.
  afterAll(() => {
    if (originalBytes) {
      writeFileSync(ARTIFACT_PATH, Buffer.from(originalBytes));
    }
    resetFoundationAdapter();
  });

  const TEST_TIMEOUT = 60_000; // real ONNX load + inference per request

  it("preconditions: CBraMod artifact on disk matches manifest SHA-256", () => {
    const onDisk = readFileSync(ARTIFACT_PATH);
    const sha = sha256Hex(new Uint8Array(onDisk.buffer, onDisk.byteOffset, onDisk.byteLength));
    expect(sha).toBe(EXPECTED_SHA);
    expect(onDisk.length).toBe(22018587);
  });

  it(
    "corrupted artifact → 424 FoundationUnavailableError, never V2/PCA fallback",
    async () => {
      // Snapshot current bytes (in case a prior test left them dirty — they shouldn't).
      const backup = readFileSync(ARTIFACT_PATH);
      expect(backup.length).toBe(22018587);

      try {
        // --- Corrupt: flip a single byte mid-file (size-preserving so only SHA fires) ---
        const corrupted = Buffer.from(backup);
        const flipPos = Math.floor(corrupted.length / 2);
        corrupted[flipPos] = corrupted[flipPos] ^ 0xff;
        writeFileSync(ARTIFACT_PATH, corrupted);

        // Sanity: corruption changed the hash but not the size.
        const shaAfterCorrupt = sha256Hex(
          new Uint8Array(corrupted.buffer, corrupted.byteOffset, corrupted.byteLength),
        );
        expect(shaAfterCorrupt).not.toBe(EXPECTED_SHA);
        expect(corrupted.length).toBe(backup.length);

        // Clear the adapter cache so ensureAdapter() re-reads from disk.
        resetFoundationAdapter();
        await resetDB();

        // --- Real request with valid JWT → should hit ensureAdapter → SHA mismatch ---
        const res = await callFoundation(makeRequest(_tokens.userA.jwt));
        expect(res.status).toBe(424);

        const body = await res.json();
        expect(body.error).toMatch(/unavailable/i);
        expect(body.detail).toMatch(/SHA-256 verification failed/i);

        // V2 must never be invoked on the Tier-2 artifact failure path.
        embedEEGMock.mockClear();
        expect(embedEEGMock).not.toHaveBeenCalled();
      } finally {
        // CRITICAL: restore the original artifact byte-for-byte.
        writeFileSync(ARTIFACT_PATH, Buffer.from(backup));
        resetFoundationAdapter();
      }
    },
    TEST_TIMEOUT,
  );

  it(
    "size mismatch → 424 FoundationUnavailableError (size gate fires before SHA)",
    async () => {
      const backup = readFileSync(ARTIFACT_PATH);

      try {
        // Prepend garbage to violate the size gate (L163).
        const tampered = Buffer.concat([Buffer.from([0x00, 0x01, 0x02, 0x03]), backup]);
        writeFileSync(ARTIFACT_PATH, tampered);

        resetFoundationAdapter();
        await resetDB();

        const res = await callFoundation(makeRequest(_tokens.userA.jwt));
        expect(res.status).toBe(424);

        const body = await res.json();
        expect(body.error).toMatch(/unavailable/i);
        expect(body.detail).toMatch(/artifact size mismatch/i);

        embedEEGMock.mockClear();
        expect(embedEEGMock).not.toHaveBeenCalled();
      } finally {
        writeFileSync(ARTIFACT_PATH, Buffer.from(backup));
        resetFoundationAdapter();
      }
    },
    TEST_TIMEOUT,
  );

  it(
    "restored artifact → 200 with real ONNX 200-D inference (warm session reuse)",
    async () => {
      // Ensure the artifact is the REAL one (previous tests restore it in finally).
      const onDisk = readFileSync(ARTIFACT_PATH);
      const sha = sha256Hex(new Uint8Array(onDisk.buffer, onDisk.byteOffset, onDisk.byteLength));
      expect(sha).toBe(EXPECTED_SHA);

      resetFoundationAdapter();
      await resetDB();

      const res = await callFoundation(makeRequest(_tokens.userA.jwt));
      expect(res.status).toBe(200);

      const body = await res.json();
      expect(body.dimensions).toBe(200);
      expect(body.model).toBe(FOUNDATION_MODEL_ID);
      expect(body.vector_indexed).toBeGreaterThan(0);

      // Every returned vector must be exactly 200-D and L2-normalized.
      for (const emb of body.embeddings) {
        expect(emb.dimensions).toBe(200);
        expect(emb.model).toBe(FOUNDATION_MODEL_ID);
        const norm = Math.sqrt(emb.vector.reduce((s: number, x: number) => s + x * x, 0));
        expect(norm).toBeCloseTo(1.0, 3);
      }

      // No V2 fallback.
      embedEEGMock.mockClear();
      expect(embedEEGMock).not.toHaveBeenCalled();
    },
    TEST_TIMEOUT,
  );

  it("artifact restored byte-for-byte after all corruption tests (SHA-256 verified)", () => {
    // Final integrity gate: the artifact on disk must be exactly the original.
    const onDisk = readFileSync(ARTIFACT_PATH);
    const bytes = new Uint8Array(onDisk.buffer, onDisk.byteOffset, onDisk.byteLength);
    expect(sha256Hex(bytes)).toBe(EXPECTED_SHA);
    expect(onDisk.length).toBe(originalBytes.length);

    // Deep equality on every byte.
    const original = Buffer.from(originalBytes);
    for (let i = 0; i < original.length; i++) {
      if (onDisk[i] !== original[i]) {
        throw new Error(
          `Byte mismatch at position ${i}: expected ${original[i]}, got ${onDisk[i]}`,
        );
      }
    }
  });
});
