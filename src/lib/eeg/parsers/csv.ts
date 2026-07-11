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
  const nanCountPerChannel = new Array<number>(C).fill(0);

  for (let r = 0; r < dataLines.length; r++) {
    const cells = split(dataLines[r]);
    for (let c = 0; c < C; c++) {
      const v = Number(cells[c]);
      if (!Number.isFinite(v)) {
        nanCount++;
        nanCountPerChannel[c]++;
        if (nanLocations.length < 10) {
          nanLocations.push(`row ${r + 1}, ch ${channels[c]}`);
        }
        data[c][r] = r > 0 && Number.isFinite(data[c][r - 1]) ? data[c][r - 1] : 0;
      } else {
        data[c][r] = v;
      }
    }
  }

  const warnings: string[] = [];
  if (nanCount > 0) {
    warnings.push(
      `${nanCount} non-finite value(s) forward-filled. Locations: ${nanLocations.join("; ")}${
        nanCount > nanLocations.length ? "; …" : ""
      }`,
    );
  }

  const nanPercent = dataLines.length * C > 0 ? (nanCount / (dataLines.length * C)) * 100 : 0;
  if (nanPercent > 20) {
    throw new Error(
      `CSV: ${nanPercent.toFixed(1)}% of values are non-finite. File may be corrupted.`,
    );
  }

  // A whole-file threshold can hide one fully-dead channel in a
  // multi-channel file (e.g. 1 of 8 channels 100% non-finite is only
  // 12.5% file-wide, well under the 20% cutoff above, and would
  // otherwise be silently forward-filled into a flat channel). Reject
  // any single channel that's mostly garbage on its own terms too.
  const CHANNEL_NAN_THRESHOLD = 50;
  const badChannels: string[] = [];
  for (let c = 0; c < C; c++) {
    const chPercent = dataLines.length > 0 ? (nanCountPerChannel[c] / dataLines.length) * 100 : 0;
    if (chPercent > CHANNEL_NAN_THRESHOLD) {
      badChannels.push(`${channels[c]} (${chPercent.toFixed(1)}%)`);
    }
  }
  if (badChannels.length > 0) {
    throw new Error(
      `CSV: channel(s) exceed ${CHANNEL_NAN_THRESHOLD}% non-finite values: ${badChannels.join(", ")}. Likely a disconnected/dead channel.`,
    );
  }

  return {
    channels,
    data,
    sampleRate,
    meta: { format: "csv", nan_count: nanCount, nan_percent: nanPercent, warnings },
  };
}
