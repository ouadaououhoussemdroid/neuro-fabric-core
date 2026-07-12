import { createFileRoute } from "@tanstack/react-router";
import { parseEDF, parseCSV, parseNPY } from "@/lib/eeg/parsers";
import { preprocess } from "@/lib/eeg/preprocessing";
import { embedSignal } from "@/lib/embeddings";
import { decodeCognitiveState } from "@/lib/decoder";
import { log, startTimer } from "@/lib/logging";
import { authenticateRequest, AuthError } from "@/integrations/supabase/request-auth";
import { checkRateLimit } from "@/integrations/supabase/rate-limit";
import type { EEGSignal } from "@/lib/eeg/types";

const MAX_FILE_BYTES = 50 * 1024 * 1024; // T-028: 50 MB cap
const ALLOWED_TYPES = [".edf", ".bdf", ".csv", ".tsv", ".npy"];

const RATE_LIMIT_MAX = 20;
const RATE_LIMIT_WINDOW_SECONDS = 60;

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
        const overall = startTimer("eeg.upload.total");
        try {
          let userId: string;
          let supabase: Awaited<ReturnType<typeof authenticateRequest>>["supabase"];
          try {
            const auth = await authenticateRequest(request);
            userId = auth.userId;
            supabase = auth.supabase;
          } catch (authErr) {
            const status = authErr instanceof AuthError ? authErr.status : 401;
            return json({ error: (authErr as Error).message ?? "Unauthorized" }, status);
          }

          let rl: { allowed: boolean; retryAfterMs: number };
          try {
            rl = await checkRateLimit(supabase, userId, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SECONDS);
          } catch (rlErr) {
            // Fail open: an unavailable rate-limit check shouldn't take down
            // the upload path, but it must be loud so a persistent outage is
            // noticed rather than silently disabling rate limiting.
            log("warn", "eeg.upload.rate_limit_check_failed", {
              error: (rlErr as Error).message,
              userId,
            });
            rl = { allowed: true, retryAfterMs: 0 };
          }
          if (!rl.allowed) {
            return json(
              { error: "Rate limit exceeded. Try again shortly.", retry_after_ms: rl.retryAfterMs },
              429,
            );
          }

          const ct = request.headers.get("content-type") ?? "";
          if (!ct.includes("multipart/form-data")) {
            return json({ error: "expected multipart/form-data" }, 400);
          }

          const form = await request.formData();
          const file = form.get("file");
          if (!(file instanceof File)) {
            return json({ error: "missing 'file' field" }, 400);
          }

          if (file.size > MAX_FILE_BYTES) {
            return json(
              {
                error: `File too large. Maximum is ${MAX_FILE_BYTES / 1024 / 1024} MB.`,
                file_size_bytes: file.size,
              },
              413,
            );
          }

          if (file.size === 0) {
            return json({ error: "Uploaded file is empty." }, 400);
          }

          const filename = file.name || "upload";
          const lower = filename.toLowerCase();
          const ext = ALLOWED_TYPES.find((e) => lower.endsWith(e));
          if (!ext) {
            return json(
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
            return json(
              { error: `File content does not match ${ext} format (magic number check failed).` },
              422,
            );
          }

          const sampleRateRaw = form.get("sampleRate");
          const latentDim = Math.min(512, Math.max(8, Number(form.get("latentDim") ?? 64)));
          const sizeBytes = file.size;

          const tUpload = startTimer("eeg.upload.parse", { filename, sizeBytes });
          let signal: EEGSignal;
          try {
            if (ext === ".edf" || ext === ".bdf") {
              signal = parseEDF(fileBuffer);
            } else if (ext === ".csv" || ext === ".tsv") {
              const fs = Number(sampleRateRaw);
              if (!Number.isFinite(fs) || fs <= 0)
                return json({ error: "sampleRate required for CSV" }, 400);
              signal = parseCSV(new TextDecoder().decode(fileBuffer), fs);
            } else if (ext === ".npy") {
              const fs = Number(sampleRateRaw);
              if (!Number.isFinite(fs) || fs <= 0)
                return json({ error: "sampleRate required for NPY" }, 400);
              signal = parseNPY(fileBuffer, fs);
            } else {
              return json({ error: `Unsupported file type: ${filename}` }, 415);
            }
          } catch (parseErr) {
            return json({ error: `Failed to parse file: ${(parseErr as Error).message}` }, 422);
          }

          const uploadMs = tUpload.end({
            channels: signal.channels.length,
            samples: signal.data[0]?.length ?? 0,
          });

          if (signal.channels.length === 0 || !signal.data[0] || signal.data[0].length === 0) {
            return json({ error: "Parsed signal has no data." }, 422);
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

          if (pre.windows.length === 0) {
            return json({ error: "Signal too short for window segmentation." }, 422);
          }

          const tEmb = startTimer("eeg.upload.embed", { filename });
          const emb = embedSignal(pre.windows, latentDim);
          const embedMs = tEmb.end({ model: emb.model, dim: emb.dimensions });

          const tDec = startTimer("eeg.upload.decode", { filename });
          const decoder = decodeCognitiveState(pre.signal);
          const decodeMs = tDec.end();
          const totalMs = overall.end({ filename });

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
                embedding_dimensions: emb.dimensions,
                embedding_model: emb.model,
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

          return json({
            analysis_id: analysisId,
            persisted,
            embedding: emb.vector,
            dimensions: emb.dimensions,
            model: emb.model,
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
        } catch (err) {
          log("error", "eeg.upload.failed", { error: (err as Error).message });
          return json({ error: "Internal server error." }, 500);
        }
      },
    },
  },
});

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function parseJsonField(v: FormDataEntryValue | null): Record<string, unknown> | null {
  if (typeof v !== "string" || v.length === 0) return null;
  try {
    return JSON.parse(v) as Record<string, unknown>;
  } catch {
    return null;
  }
}
