import type { EEGSignal } from "./types";
import { parseEDF, parseCSV, parseNPY } from "./parsers";

/**
 * T-001 — Hardware-agnostic acquisition adapter.
 *
 * A single interface through which the rest of the platform consumes EEG,
 * regardless of whether the samples originate from an uploaded file, a
 * BrainFlow board, or an LSL stream. Each concrete source produces a stream
 * of {@link EEGSignal} chunks (a whole file is one chunk; live sources emit
 * fixed-duration windows).
 *
 * The file-backed adapter is implemented here. The BrainFlow (T-005) and LSL
 * (T-004) adapters live in their respective task modules and register
 * factories against the same interface.
 */

/** Format hints for the file-backed adapter. */
export type FileFormat = "edf" | "bdf" | "csv" | "tsv" | "npy";

/** Discriminated union describing the origin of a chunk stream. */
export type SourceType = "file" | "brainflow" | "lsl";

/**
 * Contract every acquisition source satisfies.
 *
 * `stream()` is async-iterable so callers can `for await` over chunks as they
 * arrive; file sources emit exactly one chunk (the parsed recording), while
 * live sources emit successive windows until closed.
 */
export interface AcquisitionSource {
  readonly type: SourceType;
  readonly metadata: { sampleRate: number; channels: string[] };
  stream(): AsyncIterable<EEGSignal>;
  /** Release any underlying resource (board handle, socket, etc.). */
  close?(): Promise<void> | void;
}

export interface FileSourceConfig {
  /** Raw file contents. */
  data: ArrayBuffer | string;
  /** Detected/supplied format. */
  format: FileFormat;
  /** Required for CSV/TSV/NPY (samples per second). Ignored for EDF/BDF. */
  sampleRate?: number;
  /** Original filename, surfaced in chunk meta for provenance. */
  filename?: string;
}

/**
 * Build an {@link AcquisitionSource} backed by an uploaded recording.
 *
 * Dispatches to the existing parsers in `./parsers`, mirroring the extension
 * dispatch in the upload route so behaviour stays consistent.
 */
export function createFileSource(config: FileSourceConfig): AcquisitionSource {
  const { format, filename } = config;
  // Resolve once; file sources emit a single chunk.
  const parsed = (): EEGSignal => {
    if (format === "edf" || format === "bdf") {
      const buf = typeof config.data === "string" ? strToArrayBuffer(config.data) : config.data;
      return parseEDF(buf);
    }
    if (format === "csv" || format === "tsv") {
      if (typeof config.data !== "string") {
        throw new Error(`${format} source requires string input`);
      }
      const fs = config.sampleRate;
      if (fs === undefined || !Number.isFinite(fs) || fs <= 0) {
        throw new Error(`sampleRate required for ${format.toUpperCase()} source`);
      }
      return parseCSV(config.data, fs);
    }
    if (format === "npy") {
      const buf = typeof config.data === "string" ? strToArrayBuffer(config.data) : config.data;
      const fs = config.sampleRate;
      if (fs === undefined || !Number.isFinite(fs) || fs <= 0) {
        throw new Error("sampleRate required for NPY source");
      }
      return parseNPY(buf, fs);
    }
    throw new Error(`Unsupported file format: ${format}`);
  };

  let cached: EEGSignal | null = null;
  return {
    type: "file",
    get metadata() {
      cached ??= parsed();
      return { sampleRate: cached.sampleRate, channels: cached.channels };
    },
    async *stream() {
      cached ??= parsed();
      yield {
        ...cached,
        meta: { ...(cached.meta ?? {}), filename, source: "file" },
      };
    },
  };
}

function strToArrayBuffer(s: string): ArrayBuffer {
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i) & 0xff;
  return bytes.buffer;
}
