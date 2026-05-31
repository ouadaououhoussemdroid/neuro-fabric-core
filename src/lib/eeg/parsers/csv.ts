import type { EEGSignal } from "../types";

/**
 * Parse CSV EEG. Conventions:
 *   - First row may be a header. If first cell is non-numeric, it is treated as header.
 *   - Columns = channels. Rows = samples.
 *   - Sample rate must be supplied externally (no universal CSV metadata).
 */
export function parseCSV(text: string, sampleRate: number): EEGSignal {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error("CSV: sampleRate is required and must be > 0");
  }
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) throw new Error("CSV: empty");

  const split = (line: string) => line.split(/[,;\t]/).map((s) => s.trim());
  const first = split(lines[0]);
  const headerIsText = first.some((c) => c !== "" && Number.isNaN(Number(c)));
  const channels = headerIsText ? first : first.map((_, i) => `ch${i}`);
  const dataLines = headerIsText ? lines.slice(1) : lines;

  const C = channels.length;
  const data: number[][] = Array.from({ length: C }, () => new Array<number>(dataLines.length));
  for (let r = 0; r < dataLines.length; r++) {
    const cells = split(dataLines[r]);
    for (let c = 0; c < C; c++) {
      const v = Number(cells[c]);
      data[c][r] = Number.isFinite(v) ? v : 0;
    }
  }
  return { channels, data, sampleRate, meta: { format: "csv" } };
}