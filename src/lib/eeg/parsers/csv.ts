import type { EEGSignal } from "../types";

export function parseCSV(text: string, sampleRate: number): EEGSignal {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error("CSV: sampleRate is required and must be > 0");
  }

  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) throw new Error("CSV: file is empty");

  const split = (line: string) => line.split(/[,;\t]/).map((s) => s.trim());
  const first = split(lines[0]);
  const headerIsText = first.some((c) => c !== "" && Number.isNaN(Number(c)));
  const channels = headerIsText ? first : first.map((_, i) => `ch${i}`);
  const dataLines = headerIsText ? lines.slice(1) : lines;

  if (dataLines.length === 0) throw new Error("CSV: no data rows after header");

  const C = channels.length;
  const data: number[][] = Array.from({ length: C }, () => new Array<number>(dataLines.length));

  let nanCount = 0;
  const nanLocations: string[] = [];

  for (let r = 0; r < dataLines.length; r++) {
    const cells = split(dataLines[r]);
    for (let c = 0; c < C; c++) {
      const v = Number(cells[c]);
      if (!Number.isFinite(v)) {
        nanCount++;
        if (nanLocations.length < 10) {
          nanLocations.push(`row ${r + 1}, ch ${channels[c]}`);
        }
        data[c][r] = r > 0 && Number.isFinite(data[c][r - 1]) ? data[c][r - 1] : 0;
      } else {
        data[c][r] = v;
      }
    }
  }

  if (nanCount > 0) {
    console.warn(`[CSV parser] ${nanCount} non-finite value(s) interpolated. Locations: ${nanLocations.join("; ")}`);
  }

  const nanPercent = dataLines.length * C > 0 ? (nanCount / (dataLines.length * C)) * 100 : 0;
  if (nanPercent > 20) {
    throw new Error(`CSV: ${nanPercent.toFixed(1)}% of values are non-finite. File may be corrupted.`);
  }

  return {
    channels,
    data,
    sampleRate,
    meta: { format: "csv", nan_count: nanCount, nan_percent: nanPercent },
  };
}
