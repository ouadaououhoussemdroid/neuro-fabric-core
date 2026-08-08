import { createFileRoute } from "@tanstack/react-router";
import { parseEDF, parseCSV, parseNPY } from "@/lib/eeg/parsers";
import { preprocess } from "@/lib/eeg/preprocessing";
import { embedEEG } from "@/lib/ai/inference/embed-eeg";
import { decodeCognitiveState } from "@/lib/decoder";
import { log, startTimer } from "@/lib/logging";
import { authenticateRequest, AuthError } from "@/integrations/supabase/request-auth";
import { checkRateLimit } from "@/integrations/supabase/rate-limit";
import type { EEGSignal } from "@/lib/eeg/types";
import { metrics } from "@/lib/metrics";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";
import { NeuralVectorIndex, type NeuralVectorIndexOptions } from "@/lib/vector-search/neural-index";

const MAX_FILE_BYTES = 50 * 1024 * 1024; // T-028: 50 MB cap
const ALLOWED_TYPES = [".edf", ".bdf", ".csv", ".tsv", ".npy"];

const RATE_LIMIT_MAX = 20;
const RATE_LIMIT_WINDOW_SECONDS = 60;

// T-PR-002 — Maximum processing time for a single upload. Prevents a single
// slow upload from consuming server resources indefinitely.
const PROCESSING_TIMEOUT_MS = 60_000;

/**
 * T-028 — Magic-number (content sniff) checks for each allowed format.
 * Prevents a renamed .txt file from being passed to the parser, which
 * could trigger unhandled errors or DoS via large allocations.
 */
const MAGIC_NUMBERS: Record<string, number[]> = {
  // EDF/BDF: version field is "0" (0x30) for EDF, 0xFF for BDF.
  ".edf": [0x30],
  ".bdf": [0xff],
  // CSV/TSV: no reliable magic number — skip (rely on parser validation).
  ".csv": [],
  ".tsv": [],
  // NPY: starts with the numpy magic "\x93NUMPY".
  ".npy": [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59],
};

function checkMagicNumber(bytes: Uint8Array, ext: string): boolean {
  const expected = MAGIC_NUMBERS[ext];
  if (!expected || expected.length === 0) return true; // no check for this format
  if (bytes.length < expected.length) return false;
  return expected.every((b, i) => bytes[i] === b);
}

export const Route = createFileRoute("/api/eeg/upload")({
  server: {
    handlers: {
      POST: async ({ request, context }) => {
        // CORS pre-flight / origin check.
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        // Capture origin for CORS response headers on actual requests.
        const requestOrigin = request.headers.get("origin");
        const res = makeJson(requestOrigin);

        const overall = startTimer("eeg.upload.total");
        void overall;
        metrics.uploadRequestsTotal.inc();
        metrics.inFlightUploads.inc();

        // T-PR-002 — race processing against a hard 60s timeout. Prevents a
        // single slow upload from consuming server resources indefinitely.
        let timeoutId: ReturnType<typeof setTimeout>;
        const timeoutPromise = new Promise<never>((_, reject) => {
          timeoutId = setTimeout(() => {
            reject(new Error(`Upload processing timeout exceeded (${PROCESSING_TIMEOUT_MS}ms)`));
          }, PROCESSING_TIMEOUT_MS);
        });

        try {
          return await Promise.race([processUpload(), timeoutPromise]);
        } catch (err) {
          const msg = (err as Error).message;
          if (msg.includes("timeout exceeded")) {
            log("error", "eeg.upload.timeout_exceeded", { timeoutMs: PROCESSING_TIMEOUT_MS });
            return res(
              { error: "Processing timeout. Please try a smaller file or reduce channel count." },
              408,
            );
          }
          // T-PR-003 — sanitize error messages: never leak internals.
          log("error", "eeg.upload.failed", { error: msg });
          metrics.uploadErrorsTotal.inc();
          return res({ error: "An error occurred during processing." }, 500);
        } finally {
          clearTimeout(timeoutId!);
          // T-PR-008: always decrement in-flight gauge.
          metrics.inFlightUploads.dec();
        }

        async function processUpload(): Promise<Response> {
          let userId: string;
          let supabase: Awaited<ReturnType<typeof authenticateRequest>>["supabase"];
          try {
            const auth = await authenticateRequest(request);
            userId = auth.userId;
            supabase = auth.supabase;
          } catch (authErr) {
            // T-PR-003 — sanitize: only trust AuthError messages; for
            // unexpected errors, return a generic message and log the real
            // error server-side.
            if (authErr instanceof AuthError) {
              return res({ error: authErr.message }, authErr.status);
            }
            log("error", "eeg.upload.auth_unexpected", {
              error: (authErr as Error).message,
            });
            return res({ error: "Authentication failed." }, 401);
          }

          let rl: { allowed: boolean; retryAfterMs: number };
          try {
            rl = await checkRateLimit(supabase, userId, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SECONDS);
          } catch (rlErr) {
            // Fail closed: if the rate-limit RPC is unavailable we cannot
            // verify that the user is within limits, so we block the request
            // to protect the service from potential abuse during outages.
            log("error", "eeg.upload.rate_limit_check_failed", {
              error: (rlErr as Error).message,
              userId,
            });
            return res({ error: "Rate limit service unavailable. Please try again shortly." }, 503);
          }
          if (!rl.allowed) {
            metrics.rateLimitedTotal.inc();
            return res(
              { error: "Rate limit exceeded. Try again shortly.", retry_after_ms: rl.retryAfterMs },
              429,
            );
          }

          const ct = request.headers.get("content-type") ?? "";
          if (!ct.includes("multipart/form-data")) {
            return res({ error: "expected multipart/form-data" }, 400);
          }

          const form = await request.formData();
          const file = form.get("file");
          if (!(file instanceof File)) {
            return res({ error: "missing 'file' field" }, 400);
          }

          // T-008: record upload size in metrics.
          metrics.uploadBytesTotal.inc({}, file.size);

          if (file.size > MAX_FILE_BYTES) {
            return res(
              {
                error: `File too large. Maximum is ${MAX_FILE_BYTES / 1024 / 1024} MB.`,
                file_size_bytes: file.size,
              },
              413,
            );
          }

          if (file.size === 0) {
            return res({ error: "Uploaded file is empty." }, 400);
          }

          // T-028: sanitize filename to prevent path traversal (defence-in-depth, even
          // though Supabase Storage doesn't honour directory separators in names).
          const rawFilename = file.name || "upload";
          const filename = sanitizeFilename(rawFilename);
          const lower = filename.toLowerCase();
          const ext = ALLOWED_TYPES.find((e) => lower.endsWith(e));
          if (!ext) {
            return res(
              { error: `Unsupported file type: ${filename}. Allowed: ${ALLOWED_TYPES.join(", ")}` },
              415,
            );
          }

          // T-028: magic-number content sniff — read the first few bytes
          // and verify they match the expected signature for this format.
          const fileBuffer = await file.arrayBuffer();
          const head = new Uint8Array(fileBuffer.slice(0, 16));
          if (!checkMagicNumber(head, ext)) {
            log("warn", "eeg.upload.magic_number_mismatch", {
              filename,
              ext,
              headBytes: Array.from(head.slice(0, 8)).join(","),
            });
            return res(
              { error: `File content does not match ${ext} format (magic number check failed).` },
              422,
            );
          }

          const sampleRateRaw = form.get("sampleRate");
          // latentDim form field accepted for backward-compat but the canonical
          // embedding dimension is now 32 (vector(32) contract). The AI facade
          // (embedEEG) owns the dimension; do not pass latentDim to producers.
          const sizeBytes = file.size;

          const tUpload = startTimer("eeg.upload.parse", { filename, sizeBytes });
          let signal: EEGSignal;
          try {
            if (ext === ".edf" || ext === ".bdf") {
              signal = parseEDF(fileBuffer);
            } else if (ext === ".csv" || ext === ".tsv") {
              const fs = Number(sampleRateRaw);
              if (!Number.isFinite(fs) || fs <= 0)
                return res({ error: "sampleRate required for CSV" }, 400);
              signal = parseCSV(new TextDecoder().decode(fileBuffer), fs);
            } else if (ext === ".npy") {
              const fs = Number(sampleRateRaw);
              if (!Number.isFinite(fs) || fs <= 0)
                return res({ error: "sampleRate required for NPY" }, 400);
              signal = parseNPY(fileBuffer, fs);
            } else {
              return res({ error: `Unsupported file type: ${filename}` }, 415);
            }
          } catch (parseErr) {
            // T-PR-003 — sanitize parse error message to avoid leaking
            // file internals. Log the full error server-side.
            log("warn", "eeg.upload.parse_error", {
              error: (parseErr as Error).message,
              filename,
              userId,
            });
            return res(
              {
                error:
                  "Failed to parse file. The file may be corrupted or in an unsupported format.",
              },
              422,
            );
          }

          const uploadMs = tUpload.end({
            channels: signal.channels.length,
            samples: signal.data[0]?.length ?? 0,
          });
          void uploadMs;
          // T-008: record parse duration.
          metrics.uploadParseMs.observe({}, uploadMs);

          if (signal.channels.length === 0 || !signal.data[0] || signal.data[0].length === 0) {
            return res({ error: "Parsed signal has no data." }, 422);
          }

          const bp = parseJsonField(form.get("bandpass"));
          const nt = parseJsonField(form.get("notch"));

          const tPre = startTimer("eeg.upload.preprocess", { filename });
          const pre = preprocess(signal, {
            bandpass: bp ? { low: Number(bp.low), high: Number(bp.high) } : undefined,
            notch: nt ? { fc: (Number(nt.fc) === 50 ? 50 : 60) as 50 | 60 } : undefined,
          });
          const preprocessMs = tPre.end({
            steps: pre.report.steps.length,
            windows: pre.windows.length,
          });
          // T-008: record preprocess duration.
          metrics.uploadPreprocessMs.observe({}, preprocessMs);

          if (pre.windows.length === 0) {
            return res({ error: "Signal too short for window segmentation." }, 422);
          }

          const tEmb = startTimer("eeg.upload.embed", { filename });
          // Route through the AI facade: embedEEG applies the EEGConformer rollout
          // gate (OFF→PCA, CANARY→cohort, GA→EEGConformer) and PCA fallback.
          // All paths produce 32-dim vectors matching vector(32).
          const emb = await embedEEG(
            { kind: "windows", windows: pre.windows },
            { userId, normalize: true },
          );
          const embedMs = tEmb.end({ model: emb.modelId, dim: emb.dim });
          // T-008: record embed duration + fallback counter.
          metrics.uploadEmbedMs.observe({ model: emb.modelId }, embedMs);
          if (emb.fellBack) {
            metrics.embedFallbackTotal.inc({ model: emb.modelId });
          }
          metrics.embedLatencyMs.observe({ model: emb.modelId }, embedMs);

          const tDec = startTimer("eeg.upload.decode", { filename });
          const decoder = await decodeCognitiveState(pre.signal);
          const decodeMs = tDec.end();
          const totalMs = overall.end({ filename });
          // T-008: record decode duration + total duration.
          metrics.uploadDecodeMs.observe({}, decodeMs);
          metrics.uploadTotalMs.observe({}, totalMs);

          let analysisId: string | null = null;
          let persisted = false;
          try {
            const { data: insertData, error: dbErr } = await supabase
              .from("eeg_analyses")
              .insert({
                user_id: userId,
                file_name: filename,
                file_size_bytes: sizeBytes,
                sample_rate: signal.sampleRate,
                num_channels: signal.channels.length,
                num_samples: signal.data[0]?.length ?? 0,
                embedding: emb.vector,
                embedding_dimensions: emb.dim,
                embedding_model: emb.modelId,
                attention: decoder.attention,
                workload: decoder.workload,
                arousal: decoder.arousal,
                bandpass_low: bp ? Number(bp.low) : null,
                bandpass_high: bp ? Number(bp.high) : null,
                notch_frequency: nt ? (Number(nt.fc) === 50 ? 50 : 60) : null,
                processing_time_ms: totalMs,
              })
              .select("id")
              .single();
            if (dbErr) {
              log("error", "eeg.upload.persist_failed", { error: dbErr.message, userId, filename });
            } else {
              analysisId = insertData?.id ?? null;
              persisted = analysisId !== null;
            }
          } catch (e) {
            log("error", "eeg.upload.persist_exception", {
              error: (e as Error).message,
              userId,
              filename,
            });
          }

          // T-011: dual-write embedding to the pgvector-backed `embeddings`
          // table so downstream ANN search (NeuralVectorIndex) is available.
          // The vector write is best-effort — if it fails, the analysis is
          // still persisted in `eeg_analyses` and the upload succeeds, but
          // `vector_indexed` is reported honestly so callers know ANN search
          // is unavailable for this embedding.
          let vectorIndexed = false;
          let vectorError: string | undefined;
          try {
            const vectorIndex = new NeuralVectorIndex({
              supabase: supabase as unknown as NeuralVectorIndexOptions["supabase"],
              modelId: emb.modelId,
              userId: userId,
              dimensions: emb.dim,
            });
            await vectorIndex.add({
              id: analysisId ?? `upload:${filename}:${Date.now()}`,
              vector: emb.vector,
              meta: {
                file_name: filename,
                model: emb.modelId,
                dimensions: emb.dim,
                samples: signal.data[0]?.length ?? 0,
                channels: signal.channels.length,
              },
            });
            vectorIndexed = true;
          } catch (e) {
            const errMsg = (e as Error).message;
            vectorError = errMsg;
            log("warn", "eeg.upload.vector_store_failed", {
              error: errMsg,
              userId,
              filename,
            });
            metrics.vectorStoreErrorsTotal.inc({ operation: "upload_add", error: "failed" });
          }

          return res({
            analysis_id: analysisId,
            persisted,
            vector_indexed: vectorIndexed,
            vector_error: vectorError,
            embedding: emb.vector,
            dimensions: emb.dim,
            model: emb.modelId,
            embed_fell_back: emb.fellBack,
            preprocessing_report: pre.report,
            decoder,
            timings: {
              upload_ms: uploadMs,
              preprocess_ms: preprocessMs,
              embed_ms: embedMs,
              decode_ms: decodeMs,
              total_ms: totalMs,
            },
            signal: {
              channels: signal.channels,
              sampleRate: signal.sampleRate,
              samples: signal.data[0]?.length ?? 0,
            },
          });
        }
      },
    },
  },
});

/**
 * T-028 — Sanitize a filename to prevent path traversal and unsafe characters.
 * Removes directory components, replaces dangerous characters, and truncates.
 * This is defence-in-depth: Supabase Storage ignores path separators in names,
 * but we never want to pass unsafe names downstream to any filesystem APIs.
 */
export function sanitizeFilename(raw: string): string {
  // Strip any directory components (both POSIX and Windows style).
  const basename = raw.split(/[/\\]/).pop() ?? "upload";
  // Replace any character outside [a-zA-Z0-9._-] with underscore.
  const cleaned = basename.replace(/[^a-zA-Z0-9._-]/g, "_");
  // Collapse sequences of dots (prevents ".." traversal and weird names).
  const safe = cleaned.replace(/\.{2,}/g, ".");
  // Truncate to a reasonable length.
  return safe.substring(0, 255) || "upload";
}

function json(body: unknown, status = 200, origin: string | null = null): Response {
  const corsHeaders = getCorsHeadersForResponse(origin);
  return applySecurityHeaders(
    new Response(JSON.stringify(body), {
      status,
      headers: {
        "content-type": "application/json",
        ...corsHeaders,
      },
    }),
  );
}

/** Create a json() closure bound to a request origin for CORS headers. */
function makeJson(origin: string | null) {
  return (body: unknown, status = 200): Response => json(body, status, origin);
}

function parseJsonField(v: FormDataEntryValue | null): Record<string, unknown> | null {
  if (typeof v !== "string" || v.length === 0) return null;
  try {
    return JSON.parse(v) as Record<string, unknown>;
  } catch {
    return null;
  }
}
