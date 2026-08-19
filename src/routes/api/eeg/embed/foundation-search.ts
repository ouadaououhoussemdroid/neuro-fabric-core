/**
 * T-036 / Mission 13 — Tier-2 (additive) CBraMod foundation SEARCH endpoint.
 *
 * This is the SMALLEST ADDITIVE EXPERIMENT closing Mission 13's "no retrieval
 * call site" gap. It searches the isolated `foundation_embeddings` (vector(200))
 * namespace for nearest neighbours of a query embedding built from the SAME
 * CBraMod pipeline as `/api/eeg/embed/foundation`.
 *
 * BOUNDARIES (do not cross):
 *  - OPT-IN, server-native; does NOT replace V2.
 *  - Does NOT modify DEFAULT_PREFERRED / embedEEG / embeddings / vector(32) / PCA.
 *  - Never makes CBraMod the default route.
 *  - Never silently falls back to V2/PCA: runtime/artifact unavailability -> 424
 *    (FoundationUnavailableError); per-window errors -> 500. The query vector is
 *    always CBraMod 200-D.
 *  - Does NOT change the browser/WASM path (server-handled route, onnxruntime-node
 *    CPU EP via foundation.server.ts).
 *  - Does NOT retrain CBraMod.
 *
 * FLOW:
 *   parse EDF/CSV/NPY/BDF -> selectCbraModChannels(19) -> resampleSignal(250)
 *   -> preprocess({bandpass:[4,38], notch:false, segment:{4,0.5}}) [matches Mission-11]
 *   -> embedFoundationWindows (onnxruntime-node CPU EP, cbramod-encoder.onnx
 *      [1,19,1000]->[1,19,5,200] -> mean-tokens -> 200-D -> L2)
 *   -> mean-across-windows + L2 -> single 200-D query vector (refusing any
 *      non-200-D input, so we never drift into the 32-D V2 space)
 *   -> searchFoundationEmbeddings -> top-K hits from foundation_embeddings.
 *
 * On a real pgvector instance this exercises `match_foundation_embeddings` (or the
 * `_exact` variant). With no Supabase client the NeuralVectorIndex uses its
 * in-memory brute-force cosine fallback (identical metric to the RPC) — so the
 * call site is exercised in tests and local dev without a database. The RPC leg
 * itself is reported as INCONCLUSIVE when Docker/Supabase is unavailable.
 */
import { createFileRoute } from "@tanstack/react-router";
import { parseEDF, parseCSV, parseNPY } from "@/lib/eeg/parsers";
import { preprocess } from "@/lib/eeg/preprocessing";
import { selectCbraModChannels } from "@/lib/eeg/channels";
import { resampleSignal } from "@/lib/eeg/preprocessing/resample";
import {
  embedFoundationWindows,
  FoundationUnavailableError,
  FOUNDATION_MODEL_ID,
  FOUNDATION_EMBEDDING_DIM,
  FOUNDATION_SAMPLE_RATE_HZ,
  foundationProvenance,
} from "@/lib/ai/inference/foundation.server";
import { searchFoundationEmbeddings } from "@/lib/ai/retrieval/foundation-search";
import { log, startTimer } from "@/lib/logging";
import { authenticateRequest, AuthError } from "@/integrations/supabase/request-auth";
import { checkRateLimit } from "@/integrations/supabase/rate-limit";
import type { EEGSignal } from "@/lib/eeg/types";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";
import { metrics } from "@/lib/metrics";

const MAX_FILE_BYTES = 50 * 1024 * 1024; // same cap as the embed route / Tier-1
const ALLOWED_TYPES = [".edf", ".bdf", ".csv", ".tsv", ".npy"];
const RATE_LIMIT_MAX = 20;
const RATE_LIMIT_WINDOW_SECONDS = 60;
// CBraMod forwards are heavier than V2 (22 MB model, native CPU EP); allow headroom.
const PROCESSING_TIMEOUT_MS = 120_000;
const DEFAULT_K = 8;
const MAX_K = 64;

/** Magic-number (content sniff) checks, identical to the embed route / Tier-1. */
const MAGIC_NUMBERS: Record<string, number[]> = {
  ".edf": [0x30],
  ".bdf": [0xff],
  ".csv": [],
  ".tsv": [],
  ".npy": [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59],
};

function checkMagicNumber(bytes: Uint8Array, ext: string): boolean {
  const expected = MAGIC_NUMBERS[ext];
  if (!expected || expected.length === 0) return true;
  if (bytes.length < expected.length) return false;
  return expected.every((b, i) => bytes[i] === b);
}

function sanitizeFilename(raw: string): string {
  const basename = raw.split(/[/\\]/).pop() ?? "upload";
  const cleaned = basename.replace(/[^a-zA-Z0-9._-]/g, "_");
  return cleaned.replace(/\.{2,}/, ".").substring(0, 255) || "upload";
}

/** L2-normalise a flat vector to unit length. */
function l2Unit(v: number[]): number[] {
  let s = 0;
  for (const x of v) s += x * x;
  const n = Math.sqrt(s) || 1;
  return v.map((x) => x / n);
}

/** Mean across window vectors, then L2-normalise to a single query vector. */
function buildQueryVector(windowVectors: number[][]): number[] {
  const dim = windowVectors[0].length;
  const acc = new Array(dim).fill(0);
  for (const w of windowVectors) {
    for (let i = 0; i < dim; i++) acc[i] += w[i];
  }
  const mean = acc.map((s) => s / windowVectors.length);
  return l2Unit(mean);
}

function json(body: unknown, status = 200, origin: string | null = null): Response {
  return applySecurityHeaders(
    new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json", ...getCorsHeadersForResponse(origin) },
    }),
  );
}

export const Route = createFileRoute("/api/eeg/embed/foundation-search")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;
        const res = json;
        const requestOrigin = request.headers.get("origin");
        const reply = (body: unknown, status = 200) => res(body, status, requestOrigin);

        const overall = startTimer("eeg.foundation.total");
        void overall;
        metrics.foundationRequestsTotal.inc();

        let timeoutId: ReturnType<typeof setTimeout>;
        const timeoutPromise = new Promise<never>((_, reject) => {
          timeoutId = setTimeout(() => {
            reject(
              new Error(`Foundation processing timeout exceeded (${PROCESSING_TIMEOUT_MS}ms)`),
            );
          }, PROCESSING_TIMEOUT_MS);
        });

        try {
          return await Promise.race([processSearch(), timeoutPromise]);
        } catch (err) {
          const msg = (err as Error).message;
          if (err instanceof FoundationUnavailableError) {
            // Fail safe — never degrade to V2/PCA.
            return reply(
              {
                error:
                  "CBraMod foundation runtime unavailable. This endpoint requires a Node.js server runtime.",
                detail: err.reason,
              },
              424,
            );
          }
          if (msg.includes("timeout exceeded")) {
            log("error", "eeg.foundation_search.timeout_exceeded", {
              timeoutMs: PROCESSING_TIMEOUT_MS,
            });
            return reply({ error: "Processing timeout. Please try a smaller file." }, 408);
          }
          log("error", "eeg.foundation_search.failed", { error: msg });
          metrics.foundationErrorsTotal.inc();
          return reply({ error: "An error occurred during processing." }, 500);
        } finally {
          clearTimeout(timeoutId!);
        }

        async function processSearch(): Promise<Response> {
          let userId: string;
          let supabase: Awaited<ReturnType<typeof authenticateRequest>>["supabase"];
          try {
            const auth = await authenticateRequest(request);
            userId = auth.userId;
            supabase = auth.supabase;
          } catch (authErr) {
            if (authErr instanceof AuthError) {
              return reply({ error: authErr.message }, authErr.status);
            }
            log("error", "eeg.foundation_search.auth_unexpected", {
              error: (authErr as Error).message,
            });
            return reply({ error: "Authentication failed." }, 401);
          }

          let rl: { allowed: boolean; retryAfterMs: number };
          try {
            rl = await checkRateLimit(supabase, userId, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SECONDS);
          } catch (rlErr) {
            log("error", "eeg.foundation_search.rate_limit_check_failed", {
              error: (rlErr as Error).message,
              userId,
            });
            return reply(
              { error: "Rate limit service unavailable. Please try again shortly." },
              503,
            );
          }
          if (!rl.allowed) {
            return reply(
              { error: "Rate limit exceeded. Try again shortly.", retry_after_ms: rl.retryAfterMs },
              429,
            );
          }

          const ct = request.headers.get("content-type") ?? "";

          // Path A: a precomputed CBraMod 200-D query vector (JSON body).
          // Used by tests + headless callers that already embedded a query. The
          // vector MUST be 200-D — reject anything else client-side (never route a
          // 32-D vector into the Tier-1 V2 space).
          if (ct.includes("application/json")) {
            const body = (await request.json()) as { vector?: number[]; k?: number };
            const queryVector = Array.isArray(body.vector) ? body.vector : null;
            if (queryVector && queryVector.length !== FOUNDATION_EMBEDDING_DIM) {
              return reply(
                {
                  error: `Query vector must be ${FOUNDATION_EMBEDDING_DIM}-D (CBraMod). Received ${queryVector.length}-D — refusing to search the 32-D V2 namespace.`,
                  received_dim: queryVector.length,
                  expected_dim: FOUNDATION_EMBEDDING_DIM,
                },
                422,
              );
            }
            const kRaw = body.k;
            const kClamped =
              typeof kRaw === "number" && kRaw > 0 ? Math.min(kRaw, MAX_K) : DEFAULT_K;
            const out = await runSearch(queryVector, kClamped, supabase, userId, undefined);
            if (out.status === 400) return reply(out.body, 400);
            return reply(out.body);
          }

          // --- Path B: raw EEG file -> same CBraMod pipeline -> 200-D query. ---
          if (!ct.includes("multipart/form-data")) {
            return reply({ error: "expected multipart/form-data" }, 400);
          }
          const form = await request.formData();
          const file = form.get("file");
          if (!(file instanceof File)) {
            return reply({ error: "missing 'file' field" }, 400);
          }
          metrics.foundationBytesTotal.inc({}, file.size);
          if (file.size > MAX_FILE_BYTES) {
            return reply(
              {
                error: `File too large. Maximum is ${MAX_FILE_BYTES / 1024 / 1024} MB.`,
                file_size_bytes: file.size,
              },
              413,
            );
          }
          if (file.size === 0) {
            return reply({ error: "Uploaded file is empty." }, 400);
          }

          const filename = sanitizeFilename(file.name || "upload");
          const lower = filename.toLowerCase();
          const ext = ALLOWED_TYPES.find((e) => lower.endsWith(e));
          if (!ext) {
            return reply(
              {
                error: `Unsupported file type: ${filename}. Allowed: ${ALLOWED_TYPES.join(", ")}}`,
              },
              415,
            );
          }
          const fileBuffer = await file.arrayBuffer();
          const head = new Uint8Array(fileBuffer.slice(0, 16));
          if (!checkMagicNumber(head, ext)) {
            log("warn", "eeg.foundation_search.magic_number_mismatch", { filename, ext });
            return reply(
              { error: `File content does not match ${ext} format (magic number check failed).` },
              422,
            );
          }

          const sampleRateRaw = form.get("sampleRate");
          const kRaw = form.get("k");
          const kClamped =
            kRaw && !Number.isNaN(Number(kRaw)) && Number(kRaw) > 0
              ? Math.min(Number(kRaw), MAX_K)
              : DEFAULT_K;

          let signal: EEGSignal;
          try {
            if (ext === ".edf" || ext === ".bdf") {
              signal = parseEDF(fileBuffer);
            } else if (ext === ".csv" || ext === ".tsv") {
              const fs = Number(sampleRateRaw);
              if (!Number.isFinite(fs) || fs <= 0)
                return reply({ error: "sampleRate required for CSV" }, 400);
              signal = parseCSV(new TextDecoder().decode(fileBuffer), fs);
            } else {
              const fs = Number(sampleRateRaw);
              if (!Number.isFinite(fs) || fs <= 0)
                return reply({ error: "sampleRate required for NPY" }, 400);
              signal = parseNPY(fileBuffer, fs);
            }
          } catch (parseErr) {
            log("warn", "eeg.foundation_search.parse_error", {
              error: (parseErr as Error).message,
              filename,
              userId,
            });
            return reply(
              {
                error:
                  "Failed to parse file. The file may be corrupted or in an unsupported format.",
              },
              422,
            );
          }

          if (signal.channels.length === 0 || !signal.data[0] || signal.data[0].length === 0) {
            return reply({ error: "Parsed signal has no data." }, 422);
          }
          // Tier-2 channel set: CBraMod's native 19-channel 10-20 montage.
          const selected = selectCbraModChannels(signal);
          const resampled = resampleSignal(selected, FOUNDATION_SAMPLE_RATE_HZ);
          const pre = preprocess(resampled, {
            bandpass: { low: 4, high: 38 },
            notch: false,
            segment: { windowSec: 4, overlap: 0.5 },
          });
          if (pre.windows.length === 0) {
            return reply({ error: "Signal too short for CBraMod window segmentation." }, 422);
          }

          const tEmb = startTimer("eeg.foundation_search.embed");
          const embeddings = await embedFoundationWindows(pre.windows);
          tEmb.end({
            model: FOUNDATION_MODEL_ID,
            dim: FOUNDATION_EMBEDDING_DIM,
            windows: embeddings.length,
          });

          // Collapse windows -> single 200-D query vector (mean + L2). Each window
          // vector is already 200-D L2, so the mean is bounded in [-1,1] per axis.
          const query = buildQueryVector(embeddings.map((r) => r.vector));
          if (query.length !== FOUNDATION_EMBEDDING_DIM) {
            // Belt-and-suspenders: guard against any drift into V2's 32-D space.
            return reply(
              {
                error:
                  "Refusing to search with a query vector of dimension " +
                  `${query.length}; foundation namespace requires ${FOUNDATION_EMBEDDING_DIM}-D (CBraMod).`,
              },
              500,
            );
          }

          const signalMeta = {
            filename,
            selected_channels: selected.channels,
            windows: pre.windows.length,
          };
          return reply((await runSearch(query, kClamped, supabase, userId, signalMeta)).body);
        }

        async function runSearch(
          queryVector: number[] | null,
          k: number,
          supabase: Awaited<ReturnType<typeof authenticateRequest>>["supabase"],
          userId: string,
          signalMeta?: Record<string, unknown>,
        ): Promise<{ status: number; body: Record<string, unknown> }> {
          if (!queryVector) {
            return { status: 400, body: { error: "No query vector supplied." } };
          }

          let results;
          try {
            results = await searchFoundationEmbeddings(queryVector, {
              supabase: supabase as never,
              userId,
              k,
            });
          } catch (searchErr) {
            log("error", "eeg.foundation_search.vector_store_failed", {
              error: (searchErr as Error).message,
              userId,
            });
            metrics.vectorStoreErrorsTotal.inc({ operation: "foundation_search", error: "failed" });
            throw searchErr;
          }

          const totalMs = overall.end({ filename: "foundation-search" });
          const provenance = foundationProvenance();

          const body: Record<string, unknown> = {
            results,
            model: FOUNDATION_MODEL_ID,
            dimensions: FOUNDATION_EMBEDDING_DIM,
            query_vector: queryVector,
            top_k: results.length,
            namespace: "foundation_embeddings",
            match_rpc:
              results.length > 0
                ? "match_foundation_embeddings"
                : "match_foundation_embeddings_exact",
            provenance,
            timings: { total_ms: totalMs },
          };
          if (signalMeta) body.signal = signalMeta;
          return { status: 200, body };
        }
      },
    },
  },
});
