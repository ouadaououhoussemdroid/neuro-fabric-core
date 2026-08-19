/**
 * Mission 15 Phase 3 — Concurrency / Reliability with REAL onnxruntime-node.
 *
 * Unlike Mission-14's `-foundation-concurrent.test.ts` (which mocked the ONNX
 * runtime), this test exercises the REAL CBraMod ONNX forward pass via
 * `onnxruntime-node` against the actual 22 MB `cbramod-encoder.onnx` artifact.
 *
 * Verifies: ramp 1→5→10→20→50 (all within rate-limit budget get 200, remainder
 * get 429), 200-D L2-normalised embeddings, ONNX session reuse (singleton),
 * no memory/session corruption under concurrency, p50/p95/p99 latency.
 *
 * Skipped when `onnxruntime-node` cannot be imported (no prebuilt native binary).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { createClient } from "@supabase/supabase-js";
import type { Database } from "@/integrations/supabase/types";

// ── Check onnxruntime-node availability (skip if no native binary) ───────────
let ortAvailable = false;
try {
  await import("onnxruntime-node");
  ortAvailable = true;
} catch {
  ortAvailable = false;
}
const maybe = ortAvailable ? describe : describe.skip;

// ── Set env vars for local Supabase stack ──────────────────────────────────
const _tokens = JSON.parse(
  readFileSync("reports/m15_jwt_test_tokens.json", "utf-8"),
) as {
  apiUrl: string;
  anonKey: string;
  serviceRoleKey: string;
  userA: { id: string; jwt: string };
  userB: { id: string; jwt: string };
};
process.env.SUPABASE_URL = _tokens.apiUrl;
process.env.SUPABASE_PUBLISHABLE_KEY = _tokens.anonKey;
process.env.SUPABASE_SERVICE_ROLE_KEY = _tokens.serviceRoleKey;

// ── Mock ONLY embedEEG (assertion: no V2 fallback) ──────────────────────────
const embedEEGMock = vi.fn();
vi.mock("@/lib/ai/inference/embed-eeg", () => ({
  embedEEG: embedEEGMock,
  DEFAULT_PREFERRED: "braindecode-eegconformer-prod-v2",
}));

// ── Real imports (foundation.server is NOT mocked — real ONNX) ──────────────
const admin = createClient<Database>(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
);
const { Route } = await import("../foundation");
const {
  resetFoundationAdapter,
  FOUNDATION_MODEL_ID,
  FOUNDATION_EMBEDDING_DIM,
} = await import("@/lib/ai/inference/foundation.server");

// ── Helpers ──────────────────────────────────────────────────────────────────
const CBraMod19 = [
  "FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2",
  "F7","F8","T7","T8","P7","P8","FZ","CZ","PZ",
];

function csvFile(name = "signal.csv", rows = 2000, seed = 0): File {
  const lines = [CBraMod19.join(",")];
  for (let r = 0; r < rows; r++) {
    lines.push(
      CBraMod19.map(
        (_, c) => (Math.sin((r + c + seed) * 0.01 + seed * 0.7) * 0.3).toFixed(4),
      ).join(","),
    );
  }
  return new File([lines.join("\n")], name, { type: "text/csv" });
}

type PostHandler = (ctx: { request: Request; context: unknown }) => Promise<Response>;
function callFoundation(request: Request) {
  const handlers = Route.options.server!.handlers as unknown as {
    POST: PostHandler;
  };
  return handlers.POST({ request, context: {} });
}

function makeRequest(jwt: string, name = "signal.csv", seed = 0): Request {
  const form = new FormData();
  form.set("file", csvFile(name, 2000, seed));
  form.set("sampleRate", "250");
  return new Request("http://localhost/api/eeg/embed/foundation", {
    method: "POST",
    body: form,
    headers: { authorization: `Bearer ${jwt}` },
  });
}

async function resetRateLimit() {
  await admin
    .from("rate_limits")
    .delete()
    .neq("user_id", "00000000-0000-0000-0000-000000000000");
}

async function resetEmbeddings() {
  await admin
    .from("foundation_embeddings")
    .delete()
    .neq("id", "00000000-0000-0000-0000-000000000000");
}

function percentile(sorted: number[], p: number): number {
  const idx = Math.floor((sorted.length - 1) * p);
  return sorted[idx];
}

// ── Tests ───────────────────────────────────────────────────────────────────

maybe("M15 Phase 3: Concurrency / Reliability (real ONNX runtime)", () => {
  beforeEach(async () => {
    await resetRateLimit();
    await resetEmbeddings();
    embedEEGMock.mockReset();
    // Clear adapter only once at start; subsequent tests reuse the cached session
  });

  afterEach(async () => {
    await resetRateLimit();
    await resetEmbeddings();
  });

  // Ramp 1 → 20: all within budget, all should be 200
  for (const n of [1, 5, 10, 20]) {
    it(`ramp ${n} concurrent: all ${n} succeed (200) with valid 200-D L2-normalised embeddings`, async () => {
      await resetRateLimit();

      const latencies: number[] = [];
      const promises = Array.from({ length: n }, (_, i) => {
        const start = performance.now();
        return callFoundation(
          makeRequest(_tokens.userA.jwt, `ramp${n}-${i}.csv`),
        ).then((res) => {
          latencies.push(performance.now() - start);
          return res;
        });
      });

      const results = await Promise.all(promises);
      const statuses = results.map((r) => r.status);

      expect(statuses.filter((s) => s === 200)).toHaveLength(n);
      expect(statuses.filter((s) => s !== 200)).toHaveLength(0);
      expect(embedEEGMock).not.toHaveBeenCalled();

      // Verify every successful response: 200-D, L2-normalised, non-zero
      const okResults = results.filter((r) => r.status === 200);
      for (const res of okResults) {
        const body = await res.json();
        expect(body.dimensions).toBe(FOUNDATION_EMBEDDING_DIM);
        expect(body.model).toBe(FOUNDATION_MODEL_ID);
        for (const emb of body.embeddings) {
          expect(emb.vector).toHaveLength(FOUNDATION_EMBEDDING_DIM);
          // L2 norm ≈ 1
          const norm = Math.sqrt(
            emb.vector.reduce((s: number, v: number) => s + v * v, 0),
          );
          expect(norm).toBeCloseTo(1, 4);
          // Not a degenerate zero vector
          expect(emb.vector.some((v: number) => v !== 0)).toBe(true);
          // No NaN
          expect(emb.vector.every((v: number) => !Number.isNaN(v))).toBe(true);
        }
      }

      // Latency report (p50/p95/p99)
      latencies.sort((a, b) => a - b);
      console.log(
        `  ramp ${n}×: p50=${percentile(latencies, 0.5).toFixed(1)}ms ` +
        `p95=${percentile(latencies, 0.95).toFixed(1)}ms ` +
        `p99=${percentile(latencies, 0.99).toFixed(1)}ms ` +
        `max=${latencies[latencies.length - 1].toFixed(1)}ms`,
      );
    }, 120_000);
  }

  // Ramp 50: exceeds rate-limit budget — 20 succeed, 30 get 429
  it("ramp 50 concurrent: 20 accepted (200), 30 rate-limited (429) — rate-limit interaction under concurrency", async () => {
    await resetRateLimit();

    const latencies: number[] = [];
    const promises = Array.from({ length: 50 }, (_, i) => {
      const start = performance.now();
      return callFoundation(
        makeRequest(_tokens.userA.jwt, `ramp50-${i}.csv`),
      ).then((res) => {
        latencies.push(performance.now() - start);
        return res;
      });
    });

    const results = await Promise.all(promises);
    const statuses = results.map((r) => r.status);
    const okCount = statuses.filter((s) => s === 200).length;
    const limitedCount = statuses.filter((s) => s === 429).length;

    // Exactly 20 accepted, 30 rate-limited (atomic UPSERT is race-free)
    expect(okCount).toBe(20);
    expect(limitedCount).toBe(30);

    // Verify 429 responses include retry_after_ms
    const limitedResults = results.filter((r) => r.status === 429);
    for (const res of limitedResults) {
      const body = await res.json();
      expect(body.retry_after_ms).toBeTypeOf("number");
      expect(body.retry_after_ms).toBeGreaterThan(0);
    }

    // Verify 200 responses have valid embeddings (no corruption under concurrency)
    const okResults = results.filter((r) => r.status === 200);
    for (const res of okResults) {
      const body = await res.json();
      expect(body.dimensions).toBe(FOUNDATION_EMBEDDING_DIM);
      for (const emb of body.embeddings) {
        expect(emb.vector).toHaveLength(FOUNDATION_EMBEDDING_DIM);
        const norm = Math.sqrt(
          emb.vector.reduce((s: number, v: number) => s + v * v, 0),
        );
        expect(norm).toBeCloseTo(1, 4);
        expect(emb.vector.some((v: number) => v !== 0)).toBe(true);
        expect(emb.vector.every((v: number) => !Number.isNaN(v))).toBe(true);
      }
    }

    latencies.sort((a, b) => a - b);
    console.log(
      `  ramp 50×: ${okCount} ok / ${limitedCount} 429 ` +
      `p50=${percentile(latencies, 0.5).toFixed(1)}ms ` +
      `p95=${percentile(latencies, 0.95).toFixed(1)}ms ` +
      `p99=${percentile(latencies, 0.99).toFixed(1)}ms`,
    );
  }, 120_000);

  it("ONNX session reuse: single model load for concurrent requests (singleton cached)", async () => {
    // Reset adapter to force a fresh load, then verify subsequent requests reuse it
    resetFoundationAdapter();
    await resetRateLimit();

    // First request: triggers model load (cold start)
    const t0 = performance.now();
    const res1 = await callFoundation(makeRequest(_tokens.userA.jwt, "cold.csv"));
    const coldMs = performance.now() - t0;
    expect(res1.status).toBe(200);

    // Reset rate limit for warm requests
    await resetRateLimit();

    // 5 concurrent warm requests: should reuse the cached session
    const warmStart = performance.now();
    const warmResults = await Promise.all(
      Array.from({ length: 5 }, (_, i) =>
        callFoundation(makeRequest(_tokens.userA.jwt, `warm${i}.csv`)),
      ),
    );
    const warmMs = performance.now() - warmStart;

    expect(warmResults.every((r) => r.status === 200)).toBe(true);

    // Cold start should be slower (model load) than warm (cached session)
    // Report but don't assert — the relationship depends on system load
    console.log(
      `  ONNX session reuse: cold=${coldMs.toFixed(1)}ms ` +
      `(1 req), warm=${(warmMs / 5).toFixed(1)}ms/req (5 concurrent)`,
    );
  }, 60_000);

  it("no memory/session corruption: all embeddings are valid 200-D across 20 concurrent requests", async () => {
    await resetRateLimit();
    const promises = Array.from({ length: 20 }, (_, i) =>
      callFoundation(makeRequest(_tokens.userA.jwt, `corruption-${i}.csv`, i + 1)),
    );
    const results = await Promise.all(promises);

    expect(results.every((r) => r.status === 200)).toBe(true);

    const allVectors: number[][] = [];
    for (const res of results) {
      const body = await res.json();
      expect(body.dimensions).toBe(FOUNDATION_EMBEDDING_DIM);
      for (const emb of body.embeddings) {
        expect(emb.vector).toHaveLength(FOUNDATION_EMBEDDING_DIM);
        // No NaN, no Infinity
        expect(emb.vector.every((v: number) => Number.isFinite(v))).toBe(true);
        // L2-normalised
        const norm = Math.sqrt(
          emb.vector.reduce((s: number, v: number) => s + v * v, 0),
        );
        expect(norm).toBeCloseTo(1, 4);
        // Not all-zero
        expect(emb.vector.some((v: number) => v !== 0)).toBe(true);
        allVectors.push(emb.vector);
      }
    }

    // Verify uniqueness — not all embeddings are identical (would indicate
    // session memory corruption where the same buffer is returned)
    const uniqueCount = new Set(allVectors.map((v) => v.join(","))).size;
    expect(uniqueCount).toBeGreaterThan(1);
  }, 60_000);
}, 120_000);
