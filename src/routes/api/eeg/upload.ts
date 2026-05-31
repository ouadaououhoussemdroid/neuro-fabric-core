import { createFileRoute } from "@tanstack/react-router";
import { parseEDF, parseCSV, parseNPY } from "@/lib/eeg/parsers";
import { preprocess } from "@/lib/eeg/preprocessing";
import { embedSignal } from "@/lib/embeddings";
import { decodeCognitiveState } from "@/lib/decoder";
import { log, startTimer } from "@/lib/logging";
import type { EEGSignal } from "@/lib/eeg/types";

/**
 * POST /api/eeg/upload
 *
 * Accepts multipart/form-data with:
 *   file:        EDF | CSV | NumPy (.npy)
 *   sampleRate:  required for CSV / NPY (number, Hz)
 *   bandpass:    optional JSON {"low":1,"high":40}
 *   notch:       optional JSON {"fc":60} (50 | 60)
 *   latentDim:   optional number (default 64)
 *
 * Returns:
 *   {
 *     embedding: number[],
 *     dimensions: number,
 *     preprocessing_report: {...},
 *     decoder: { attention, workload, arousal, ... },
 *     timings: { upload_ms, preprocess_ms, embed_ms, decode_ms }
 *   }
 */
export const Route = createFileRoute("/api/eeg/upload")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const overall = startTimer("eeg.upload.total");
        try {
          const ct = request.headers.get("content-type") ?? "";
          if (!ct.includes("multipart/form-data")) {
            return json({ error: "expected multipart/form-data" }, 400);
          }
          const form = await request.formData();
          const file = form.get("file");
          if (!(file instanceof File)) {
            return json({ error: "missing 'file' field" }, 400);
          }
          const sampleRateRaw = form.get("sampleRate");
          const latentDim = Number(form.get("latentDim") ?? 64);
          const filename = file.name || "upload";
          const sizeBytes = file.size;

          const tUpload = startTimer("eeg.upload.parse", { filename, sizeBytes });
          const lower = filename.toLowerCase();
          let signal: EEGSignal;
          if (lower.endsWith(".edf") || lower.endsWith(".bdf")) {
            signal = parseEDF(await file.arrayBuffer());
          } else if (lower.endsWith(".csv") || lower.endsWith(".tsv")) {
            const fs = Number(sampleRateRaw);
            if (!Number.isFinite(fs) || fs <= 0) {
              return json({ error: "sampleRate required for CSV" }, 400);
            }
            signal = parseCSV(await file.text(), fs);
          } else if (lower.endsWith(".npy")) {
            const fs = Number(sampleRateRaw);
            if (!Number.isFinite(fs) || fs <= 0) {
              return json({ error: "sampleRate required for NPY" }, 400);
            }
            signal = parseNPY(await file.arrayBuffer(), fs);
          } else {
            return json({ error: `unsupported file type: ${filename}` }, 415);
          }
          const uploadMs = tUpload.end({
            channels: signal.channels.length,
            samples: signal.data[0]?.length ?? 0,
            sampleRate: signal.sampleRate,
          });

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
            return json({ error: "signal too short for window segmentation" }, 422);
          }

          const tEmb = startTimer("eeg.upload.embed", { filename });
          const emb = embedSignal(pre.windows, latentDim);
          const embedMs = tEmb.end({ model: emb.model, dim: emb.dimensions });

          const tDec = startTimer("eeg.upload.decode", { filename });
          const decoder = decodeCognitiveState(pre.signal);
          const decodeMs = tDec.end();

          const totalMs = overall.end({ filename });

          return json({
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
          return json({ error: (err as Error).message }, 500);
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