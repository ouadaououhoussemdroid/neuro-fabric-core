/**
 * Shared EEG types.
 *
 * Signal shape convention:
 *   - channels: string[] of length C
 *   - data: number[][] shape [C][N] (microvolts, float32-equivalent)
 *   - sampleRate: Hz
 */
export interface EEGSignal {
  channels: string[];
  data: number[][]; // [C][N]
  sampleRate: number; // Hz
  meta?: Record<string, unknown>;
}

export interface EEGWindow {
  data: number[][]; // [C][W]
  sampleRate: number;
  start: number; // sample index in source
  end: number; // exclusive
}

export interface PreprocessingReport {
  channels: number;
  samples: number;
  sampleRate: number;
  steps: Array<{ name: string; params: Record<string, unknown>; durationMs: number }>;
  totalDurationMs: number;
}
