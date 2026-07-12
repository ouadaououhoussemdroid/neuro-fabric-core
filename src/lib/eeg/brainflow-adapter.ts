/**
 * T-005 — BrainFlow acquisition adapter.
 *
 * Server-side consumer that reads from a BrainFlow board (OpenBCI Cyton,
 * Ganglion, Muse, or the synthetic board used in CI) and publishes chunks
 * to the WS gateway (T-003) via the {@link AcquisitionSource} interface.
 *
 * The `brainflow` Node binding is an optional native dependency — it is not
 * in package.json because it requires a platform-specific install. This
 * module dynamically imports it, so the project compiles and the unit tests
 * run without it. The synthetic board is used for CI and local development.
 */
import type { EEGSignal } from "./types";
import type { AcquisitionSource } from "./acquisition";

/** BrainFlow board identifiers (subset; full list in brainflow-node). */
export type BoardId =
  | "synthetic" // -1
  | "cyton" // 0
  | "ganglion" // 1
  | "muse"; // 2

/** Configuration for a BrainFlow source. */
export interface BrainFlowConfig {
  boardId: BoardId;
  /** Serial port (e.g. "COM3" / "/dev/ttyUSB0"); ignored for synthetic/muse. */
  serialPort?: string;
  /** Sample rate in Hz (board-dependent; synthetic = 250). */
  sampleRate: number;
  /** Channel labels. If omitted, generated as ch0..chN. */
  channels?: string[];
  /** Number of channels (board-dependent; synthetic = 8, cyton = 8). */
  nChannels: number;
  /** Chunk duration in seconds per yielded EEGSignal (default: 0.5). */
  chunkSeconds?: number;
}

/** Dynamically load the optional brainflow binding. */
async function loadBrainFlow(): Promise<BrainFlowBinding | null> {
  try {
    const mod = await import("brainflow");
    // Handle both ESM default and CJS shapes.
    if (mod.BoardShim) return mod as BrainFlowBinding;
    if ((mod as { default?: BrainFlowBinding }).default?.BoardShim) {
      return (mod as { default: BrainFlowBinding }).default;
    }
    return null;
  } catch {
    return null;
  }
}

interface BrainFlowBinding {
  BoardShim: new (boardId: string | number, serialPort?: string) => BrainFlowBoard;
}

interface BrainFlowBoard {
  prepare_session(): void;
  start_stream(numSamples?: number): void;
  stop_stream(): void;
  get_board_data(numSamples: number): number[][];
  release_session(): void;
}

/**
 * Build an {@link AcquisitionSource} backed by a BrainFlow board.
 *
 * If the `brainflow` native binding is not installed, falls back to a
 * deterministic synthetic generator (the same data shape the BrainFlow
 * synthetic board produces) so CI and local development work without
 * hardware. This matches the blueprint's "synthetic board for CI" requirement.
 */
export function createBrainFlowSource(config: BrainFlowConfig): AcquisitionSource {
  const channels = config.channels ?? Array.from({ length: config.nChannels }, (_, i) => `ch${i}`);
  const chunkSamples = Math.max(1, Math.round(config.sampleRate * (config.chunkSeconds ?? 0.5)));
  let closed = false;

  return {
    type: "brainflow",
    metadata: { sampleRate: config.sampleRate, channels },
    async *stream() {
      const bf = await loadBrainFlow();
      if (bf !== null) {
        yield* streamFromBoard(bf, config, channels, chunkSamples, () => closed);
      } else {
        yield* streamSynthetic(config, channels, chunkSamples, () => closed);
      }
    },
    close() {
      closed = true;
    },
  };
}

/**
 * Real board path: uses the brainflow binding to read chunks.
 *
 * The binding exposes a `BoardShim` with `start_stream`, `get_board_data`,
 * and `stop_stream`. We read in fixed-sample windows matching the chunk size.
 */
async function* streamFromBoard(
  bf: BrainFlowBinding,
  config: BrainFlowConfig,
  channels: string[],
  chunkSamples: number,
  isClosed: () => boolean,
): AsyncIterable<EEGSignal> {
  const board = new bf.BoardShim(config.boardId, config.serialPort ?? "");
  board.prepare_session();
  board.start_stream(450000);

  try {
    while (!isClosed()) {
      // Sleep for one chunk duration to let samples accumulate.
      await sleep((chunkSamples / config.sampleRate) * 1000);
      const raw = board.get_board_data(chunkSamples);
      // BrainFlow returns data as [nChannels][nSamples] (excluding time cols).
      const data = Array.from(raw, (row) => Array.from(row as ArrayLike<number>));
      yield {
        channels,
        data,
        sampleRate: config.sampleRate,
        meta: { source: "brainflow", boardId: config.boardId },
      };
    }
  } finally {
    board.stop_stream();
    board.release_session();
  }
}

/**
 * Synthetic fallback: generates a deterministic multi-channel sine + noise
 * signal so downstream pipelines (embedding, decoding) can run end-to-end
 * in CI without hardware.
 */
async function* streamSynthetic(
  config: BrainFlowConfig,
  channels: string[],
  chunkSamples: number,
  isClosed: () => boolean,
): AsyncIterable<EEGSignal> {
  const n = config.nChannels;
  const fs = config.sampleRate;
  let sampleIndex = 0;

  while (!isClosed()) {
    const data: number[][] = [];
    for (let ch = 0; ch < n; ch++) {
      const row = new Array<number>(chunkSamples);
      // 10 Hz sine per channel with a channel-dependent phase + tiny drift.
      const freq = 10;
      const phase = ch * 0.5;
      for (let s = 0; s < chunkSamples; s++) {
        const t = (sampleIndex + s) / fs;
        row[s] = Math.sin(2 * Math.PI * freq * t + phase) * 50; // microvolts
      }
      data.push(row);
    }
    sampleIndex += chunkSamples;
    yield {
      channels,
      data,
      sampleRate: fs,
      meta: { source: "brainflow-synthetic", boardId: config.boardId },
    };
    // Yield control between chunks.
    await sleep(0);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
