/**
 * T-036 / Mission 12 — Tier-2 CBraMod foundation embed endpoint.
 *
 * This is an OPT-IN, server-native replacement representation path. It does NOT
 * touch the Tier-1 V2 pipeline: it never imports or calls `embedEEG`, never
 * writes to `eeg_analyses` or the `embeddings` table (vector(32)), never routes
 * through `DEFAULT_PREFERRED`/rollout, and never falls back to V2 or PCA. If the
 * native CBraMod runtime / artifact is unavailable it fails safe with 424;
 * per-window errors return 500.
 *
 * Flow: parse EDF/CSV/NPY → selectCbraModChannels(19) → resampleSignal(250) →
 * preprocess({bandpass:[4,38], notch:false, segment:{4,0.5}}) [matches Mission-11
 * preprocess_window: bandpass + zscore, no notch] → embedFoundationWindows
 * (onnxruntime-node CPU EP, cbramod-encoder.onnx [1,19,1000]→[1,19,5,200] →
 * mean-tokens → 200-D → L2) → write foundation_embeddings(vector(200)).
 */
import { createFileRoute } from "@tanstack/react-router";
import { randomUUID } from "node:crypto";
import { parseEDF, parseCSV, parseNPY } from "@/lib/eeg/parsers";
import { preprocess } from "@/lib/eeg/preprocessing";
import { selectCbraModChannels, selectEEGPTChannels, selectProdChannels, PROD_CHANNEL_COUNT, EEGPT_CHANNEL_COUNT } from "@/lib/eeg/channels";
import { resampleSignal } from "@/lib/eeg/preprocessing/resample";
import {
  embedFoundationWindows,
  FoundationUnavailableError,
  FOUNDATION_MODEL_ID,
  FOUNDATION_EMBEDDING_DIM,
  FOUNDATION_SAMPLE_RATE_HZ,
  foundationProvenance,
} from "@/lib/ai/inference/foundation.server";
import {
  embedJointWindows,
  embedJoint2312Windows,
  JOINT_MODEL_ID,
  JOINT_EMBEDDING_DIM,
  JOINT_2312_MODEL_ID,
  JOINT_2312_EMBEDDING_DIM,
  jointProvenance,
  joint2312Provenance,
} from "@/lib/ai/inference/joint.server";
import { log, startTimer } from "@/lib/logging";
import { authenticateRequest, AuthError } from "@/integrations/supabase/request-auth";
import { checkRateLimit } from "@/integrations/supabase/rate-limit";
import type { EEGSignal } from "@/lib/eeg/types";
import { NeuralVectorIndex, type NeuralVectorIndexOptions } from "@/lib/vector-search/neural-index";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";
import { metrics } from "@/lib/metrics";

const MAX_FILE_BYTES = 50 * 1024 * 1024; // T-028: 50 MB cap (same as Tier-1)
const ALLOWED_TYPES = [".edf", ".bdf", ".csv", ".tsv", ".npy"];
const RATE_LIMIT_MAX = 20;
const RATE_LIMIT_WINDOW_SECONDS = 60;
// CBraMod forwards are heavier than V2 (22 MB model, native CPU EP) and each
// upload embeds multiple 4 s windows; allow more headroom than Tier-1's 60s.
const PROCESSING_TIMEOUT_MS = 120_000;

/** T-028 — Magic-number (content sniff) checks, identical to Tier-1. */
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
  return cleaned.replace(/\.{2,}/g, ".").substring(0, 255) || "upload";
}

function parseJsonField(v: FormDataEntryValue | null): Record<string, unknown> | null {
  if (typeof v !== "string" || v.length === 0) return null;
  try {
    return JSON.parse(v) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export const Route = createFileRoute("/api/eeg/embed/foundation")({
  server: {
    handlers: {
      POST: async ({ request, context }) => {
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        const requestOrigin = request.headers.get("origin");
        const res = makeJson(requestOrigin);

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
          return await Promise.race([processFoundation(), timeoutPromise]);
        } catch (err) {
          const msg = (err as Error).message;
          if (err instanceof FoundationUnavailableError) {
            // T-036 / M25: foundation runtime/artifact unavailable → fail safe,
            // never degrade to V2/PCA. Callers can retry on a Node runtime.
            return res(
              {
                error:
                  "Foundation embedding runtime unavailable. This endpoint requires a Node.js server runtime.",
                detail: err.reason,
              },
              424,
            );
          }
          if (msg.includes("timeout exceeded")) {
            log("error", "eeg.foundation.timeout_exceeded", { timeoutMs: PROCESSING_TIMEOUT_MS });
            return res({ error: "Processing timeout. Please try a smaller file." }, 408);
          }
          // T-PR-003 — sanitize: never leak internals.
          log("error", "eeg.foundation.failed", { error: msg });
          metrics.foundationErrorsTotal.inc();
          return res({ error: "An error occurred during processing." }, 500);
        } finally {
          clearTimeout(timeoutId!);
        }

        async function processFoundation(): Promise<Response> {
          let userId: string;
          let supabase: Awaited<ReturnType<typeof authenticateRequest>>["supabase"];
          try {
            const auth = await authenticateRequest(request);
            userId = auth.userId;
            supabase = auth.supabase;
          } catch (authErr) {
            if (authErr instanceof AuthError) {
              return res({ error: authErr.message }, authErr.status);
            }
            log("error", "eeg.foundation.auth_unexpected", {
              error: (authErr as Error).message,
            });
            return res({ error: "Authentication failed." }, 401);
          }

          let rl: { allowed: boolean; retryAfterMs: number };
          try {
            rl = await checkRateLimit(supabase, userId, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SECONDS);
          } catch (rlErr) {
            log("error", "eeg.foundation.rate_limit_check_failed", {
              error: (rlErr as Error).message,
              userId,
            });
            return res({ error: "Rate limit service unavailable. Please try again shortly." }, 503);
          }
          if (!rl.allowed) {
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

          metrics.foundationBytesTotal.inc({}, file.size);
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

          const fileBuffer = await file.arrayBuffer();
          const head = new Uint8Array(fileBuffer.slice(0, 16));
          if (!checkMagicNumber(head, ext)) {
            log("warn", "eeg.foundation.magic_number_mismatch", { filename, ext });
            return res(
              { error: `File content does not match ${ext} format (magic number check failed).` },
              422,
            );
          }

          const sampleRateRaw = form.get("sampleRate");
          const sizeBytes = file.size;

          const tParse = startTimer("eeg.foundation.parse", { filename, sizeBytes });
          let signal: EEGSignal;
          try {
            if (ext === ".edf" || ext === ".bdf") {
              signal = parseEDF(fileBuffer);
            } else if (ext === ".csv" || ext === ".tsv") {
              const fs = Number(sampleRateRaw);
              if (!Number.isFinite(fs) || fs <= 0) {
                return res({ error: "sampleRate required for CSV" }, 400);
              }
              signal = parseCSV(new TextDecoder().decode(fileBuffer), fs);
            } else {
              const fs = Number(sampleRateRaw);
              if (!Number.isFinite(fs) || fs <= 0) {
                return res({ error: "sampleRate required for NPY" }, 400);
              }
              signal = parseNPY(fileBuffer, fs);
            }
          } catch (parseErr) {
            log("warn", "eeg.foundation.parse_error", {
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
          const parseMs = tParse.end({
            channels: signal.channels.length,
            samples: signal.data[0]?.length ?? 0,
          });

          if (signal.channels.length === 0 || !signal.data[0] || signal.data[0].length === 0) {
            return res({ error: "Parsed signal has no data." }, 422);
          }

          // Parse the `model` query param to select the embedding path.
          //   cbramod-200 (default) → 200-D CBraMod foundation embedding
          //   joint-264             → fused 264-D [CBraMod⊕V2⊕PCA] embedding (M25)
          //   joint-2312            → fused 2312-D [CBraMod⊕V2⊕PCA⊕EEGPT] embedding (M28)
          const embeddingUrl = new URL(request.url, "http://localhost");
          const embeddingModel = embeddingUrl.searchParams.get("model") ?? "cbramod-200";

          if (embeddingModel === "joint-264") {
            // M25 — Joint 264-D fusion: select both channel sets, preprocess
            // both, embed each, then fuse with fixed block weights.
            const selected19 = selectCbraModChannels(signal);
            const resampled19 = resampleSignal(selected19, FOUNDATION_SAMPLE_RATE_HZ);
            const tPre19 = startTimer("eeg.joint.preprocess_cbramod", { filename });
            const pre19 = preprocess(resampled19, {
              bandpass: { low: 4, high: 38 },
              notch: false,
              segment: { windowSec: 4, overlap: 0.5 },
            });
            tPre19.end({ windows: pre19.windows.length });

            const selected22 = selectProdChannels(signal);
            const resampled22 = resampleSignal(selected22, FOUNDATION_SAMPLE_RATE_HZ);
            const tPre22 = startTimer("eeg.joint.preprocess_v2", { filename });
            const pre22 = preprocess(resampled22, {
              bandpass: { low: 4, high: 38 },
              notch: false,
              segment: { windowSec: 4, overlap: 0.5 },
            });
            const preprocessMs = tPre22.end({ windows: pre22.windows.length });

            if (pre19.windows.length === 0 || pre22.windows.length === 0) {
              return res({ error: "Signal too short for joint window segmentation." }, 422);
            }
            if (pre19.windows.length !== pre22.windows.length) {
              return res(
                {
                  error: "Window count mismatch between 19-channel and 22-channel selections.",
                },
                422,
              );
            }

            const tEmb = startTimer("eeg.joint.embed", { filename });
            const embeddings = await embedJointWindows(pre19.windows, pre22.windows);
            const embedMs = tEmb.end({
              model: JOINT_MODEL_ID,
              dim: JOINT_EMBEDDING_DIM,
              windows: embeddings.length,
            });
            // Per-window latency observed inside embedJointWindows (precise).

            const totalMs = overall.end({ filename });

            // Persist every window embedding to the joint namespace.
            let vectorIndexed = 0;
            let vectorError: string | undefined;
            try {
              const idx = new NeuralVectorIndex({
                supabase: supabase as unknown as NeuralVectorIndexOptions["supabase"],
                tableName: "joint_embeddings",
                matchRpc: "match_joint_embeddings",
                matchRpcExact: "match_joint_embeddings_exact",
                modelId: JOINT_MODEL_ID,
                userId,
                dimensions: JOINT_EMBEDDING_DIM,
              });
              for (let i = 0; i < embeddings.length; i++) {
                const r = embeddings[i];
                const w = pre19.windows[i]; // time-aligned with pre22
                await idx.add({
                  id: randomUUID(),
                  vector: r.vector,
                  meta: {
                    file_name: filename,
                    window_index: i,
                    start: w.start,
                    end: w.end,
                    channels: PROD_CHANNEL_COUNT,
                    dimensions: r.dim,
                    model: JOINT_MODEL_ID,
                  },
                });
                vectorIndexed += 1;
              }
            } catch (e) {
              vectorError = (e as Error).message;
              log("warn", "eeg.joint.vector_store_failed", {
                error: vectorError,
                userId,
                filename,
                indexed: vectorIndexed,
              });
              metrics.vectorStoreErrorsTotal.inc({ operation: "joint_add", error: "failed" });
            }

            const provenance = jointProvenance();

            return res({
              embeddings: embeddings.map((r, i) => ({
                window_index: i,
                vector: r.vector,
                dimensions: r.dim,
                model: r.modelId,
              })),
              dimensions: JOINT_EMBEDDING_DIM,
              model: JOINT_MODEL_ID,
              channels: PROD_CHANNEL_COUNT,
              windows: embeddings.length,
              vector_indexed: vectorIndexed,
              vector_error: vectorError,
              preprocessing_report: pre19.report,
              signal: {
                channels: signal.channels,
                selected_channels_19: selected19.channels,
                selected_channels_22: selected22.channels,
                sampleRate: signal.sampleRate,
                samples: signal.data[0]?.length ?? 0,
              },
              provenance,
              timings: {
                parse_ms: parseMs,
                preprocess_ms: preprocessMs,
                embed_ms: embedMs,
                total_ms: totalMs,
              },
            });
          }

          // M28 — Joint-2312 4-block fusion: CBraMod-200 ⊕ V2-32 ⊕ PCA-32 ⊕ EEGPT-2048 → 2312-D.
          // M27 proved this significantly improves retrieval: R@5=0.8527 vs Joint-264 0.7858
          // (Δ=+0.0669, p=4.8e-28, Cohen's d=0.704). Uses M27-learned block weights
          // [0.3062, 0.1434, 0.1519, 0.3985] (stable across 50 LOSO folds, CV < 0.5%).
          if (embeddingModel === "joint-2312") {
            const selected19 = selectCbraModChannels(signal);
            const resampled19 = resampleSignal(selected19, FOUNDATION_SAMPLE_RATE_HZ);
            const tPre19 = startTimer("eeg.joint2312.preprocess_cbramod", { filename });
            const pre19 = preprocess(resampled19, {
              bandpass: { low: 4, high: 38 },
              notch: false,
              segment: { windowSec: 4, overlap: 0.5 },
            });
            tPre19.end({ windows: pre19.windows.length });

            const selected22 = selectProdChannels(signal);
            const resampled22 = resampleSignal(selected22, FOUNDATION_SAMPLE_RATE_HZ);
            const tPre22 = startTimer("eeg.joint2312.preprocess_v2", { filename });
            const pre22 = preprocess(resampled22, {
              bandpass: { low: 4, high: 38 },
              notch: false,
              segment: { windowSec: 4, overlap: 0.5 },
            });
            const preprocessMs = tPre22.end({ windows: pre22.windows.length });

            // EEGPT: 62-channel 10-20 montage, bandpass 1-40 Hz (M26/M27 validated)
            const selected62 = selectEEGPTChannels(signal);
            const resampled62 = resampleSignal(selected62, FOUNDATION_SAMPLE_RATE_HZ);
            const tPre62 = startTimer("eeg.joint2312.preprocess_eegpt", { filename });
            const pre62 = preprocess(resampled62, {
              bandpass: { low: 1, high: 40 },
              notch: false,
              segment: { windowSec: 4, overlap: 0.5 },
            });
            tPre62.end({ windows: pre62.windows.length });

            if (pre19.windows.length === 0 || pre22.windows.length === 0 || pre62.windows.length === 0) {
              return res({ error: "Signal too short for joint-2312 window segmentation." }, 422);
            }
            if (pre19.windows.length !== pre22.windows.length || pre19.windows.length !== pre62.windows.length) {
              return res(
                {
                  error: "Window count mismatch between 19-channel (CBraMod), 22-channel (V2/PCA), " +
                    "and 62-channel (EEGPT) selections.",
                },
                422,
              );
            }

            const tEmb = startTimer("eeg.joint2312.embed", { filename });
            const embeddings = await embedJoint2312Windows(pre19.windows, pre22.windows, pre62.windows);
            const embedMs = tEmb.end({
              model: JOINT_2312_MODEL_ID,
              dim: JOINT_2312_EMBEDDING_DIM,
              windows: embeddings.length,
            });

            const totalMs = overall.end({ filename });

            // Persist every window embedding to the joint-2312 namespace.
            let vectorIndexed = 0;
            let vectorError: string | undefined;
            try {
              const idx = new NeuralVectorIndex({
                supabase: supabase as unknown as NeuralVectorIndexOptions["supabase"],
                tableName: "joint_embeddings_2312",
                matchRpc: "match_joint_embeddings_2312",
                matchRpcExact: "match_joint_embeddings_2312_exact",
                modelId: JOINT_2312_MODEL_ID,
                userId,
                dimensions: JOINT_2312_EMBEDDING_DIM,
              });
              for (let i = 0; i < embeddings.length; i++) {
                const r = embeddings[i];
                const w = pre19.windows[i]; // time-aligned with pre22 and pre62
                await idx.add({
                  id: randomUUID(),
                  vector: r.vector,
                  meta: {
                    file_name: filename,
                    window_index: i,
                    start: w.start,
                    end: w.end,
                    channels: EEGPT_CHANNEL_COUNT,
                    cbramod_channels: 19,
                    v2_channels: 22,
                    eegpt_channels: 62,
                    dimensions: r.dim,
                    model: JOINT_2312_MODEL_ID,
                  },
                });
                vectorIndexed += 1;
              }
            } catch (e) {
              vectorError = (e as Error).message;
              log("warn", "eeg.joint2312.vector_store_failed", {
                error: vectorError,
                userId,
                filename,
                indexed: vectorIndexed,
              });
              metrics.vectorStoreErrorsTotal.inc({ operation: "joint2312_add", error: "failed" });
            }

            const provenance = joint2312Provenance();

            return res({
              embeddings: embeddings.map((r, i) => ({
                window_index: i,
                vector: r.vector,
                dimensions: r.dim,
                model: r.modelId,
              })),
              dimensions: JOINT_2312_EMBEDDING_DIM,
              model: JOINT_2312_MODEL_ID,
              channels: EEGPT_CHANNEL_COUNT,
              windows: embeddings.length,
              vector_indexed: vectorIndexed,
              vector_error: vectorError,
              preprocessing_report: pre19.report,
              signal: {
                channels: signal.channels,
                selected_channels_19: selected19.channels,
                selected_channels_22: selected22.channels,
                selected_channels_62: selected62.channels,
                sampleRate: signal.sampleRate,
                samples: signal.data[0]?.length ?? 0,
              },
              provenance,
              timings: {
                parse_ms: parseMs,
                preprocess_ms: preprocessMs,
                embed_ms: embedMs,
                total_ms: totalMs,
              },
            });
          }

          // Tier-2 channel set: CBraMod's native 19-channel 10-20 montage.
          // Throws if any of the 19 required channels are absent — fail loud,
          // never silently zero-pad (would corrupt the 200-D representation).
          const selected = selectCbraModChannels(signal);

          // Mission-11 resamples PhysioNet 160 Hz → 250 Hz before bandpass; mirror
          // it so the segmenter yields 1000-sample windows at the model's rate.
          const resampled = resampleSignal(selected, FOUNDATION_SAMPLE_RATE_HZ);

          const tPre = startTimer("eeg.foundation.preprocess", { filename });
          // Match Mission-11 preprocess_window exactly: bandpass [4,38] + z-score,
          // 4 s windows (1000 samples @ 250 Hz), 50% overlap. No notch (M11 did
          // not apply one — `notch:false` keeps the contract faithful).
          const pre = preprocess(resampled, {
            bandpass: { low: 4, high: 38 },
            notch: false,
            segment: { windowSec: 4, overlap: 0.5 },
          });
          const preprocessMs = tPre.end({
            steps: pre.report.steps.length,
            windows: pre.windows.length,
          });

          if (pre.windows.length === 0) {
            return res({ error: "Signal too short for CBraMod window segmentation." }, 422);
          }

          const tEmb = startTimer("eeg.foundation.embed", { filename });
          const embeddings = await embedFoundationWindows(pre.windows);
          const embedMs = tEmb.end({
            model: FOUNDATION_MODEL_ID,
            dim: FOUNDATION_EMBEDDING_DIM,
            windows: embeddings.length,
          });
          // Per-window latency is observed inside embedFoundationWindows (precise);
          // `embedMs` here is the aggregate for the timings block below.

          const totalMs = overall.end({ filename });

          // Persist every window embedding to the isolated foundation namespace.
          let vectorIndexed = 0;
          let vectorError: string | undefined;
          try {
            const idx = new NeuralVectorIndex({
              supabase: supabase as unknown as NeuralVectorIndexOptions["supabase"],
              tableName: "foundation_embeddings",
              matchRpc: "match_foundation_embeddings",
              matchRpcExact: "match_foundation_embeddings_exact",
              modelId: FOUNDATION_MODEL_ID,
              userId,
              dimensions: FOUNDATION_EMBEDDING_DIM,
            });
            for (let i = 0; i < embeddings.length; i++) {
              const r = embeddings[i];
              const w = pre.windows[i];
              await idx.add({
                id: randomUUID(),
                vector: r.vector,
                meta: {
                  file_name: filename,
                  window_index: i,
                  start: w.start,
                  end: w.end,
                  channels: r.dim,
                  dimensions: r.dim,
                  model: FOUNDATION_MODEL_ID,
                },
              });
              vectorIndexed += 1;
            }
          } catch (e) {
            vectorError = (e as Error).message;
            log("warn", "eeg.foundation.vector_store_failed", {
              error: vectorError,
              userId,
              filename,
              indexed: vectorIndexed,
            });
            metrics.vectorStoreErrorsTotal.inc({ operation: "foundation_add", error: "failed" });
          }

          const provenance = foundationProvenance();

          return res({
            embeddings: embeddings.map((r, i) => ({
              window_index: i,
              vector: r.vector,
              dimensions: r.dim,
              model: r.modelId,
            })),
            dimensions: FOUNDATION_EMBEDDING_DIM,
            model: FOUNDATION_MODEL_ID,
            windows: embeddings.length,
            vector_indexed: vectorIndexed,
            vector_error: vectorError,
            preprocessing_report: pre.report,
            signal: {
              channels: signal.channels,
              selected_channels: selected.channels,
              sampleRate: signal.sampleRate,
              samples: signal.data[0]?.length ?? 0,
            },
            provenance,
            timings: {
              parse_ms: parseMs,
              preprocess_ms: preprocessMs,
              embed_ms: embedMs,
              total_ms: totalMs,
            },
          });
        }
      },
    },
  },
});

function json(body: unknown, status = 200, origin: string | null = null): Response {
  return applySecurityHeaders(
    new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json", ...getCorsHeadersForResponse(origin) },
    }),
  );
}

/** Create a json() closure bound to a request origin for CORS headers. */
function makeJson(origin: string | null) {
  return (body: unknown, status = 200): Response => json(body, status, origin);
}
