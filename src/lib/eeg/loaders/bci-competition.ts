import { parseEDF } from "../parsers/edf";
import type { EEGSignal } from "../types";
import type { DatasetLoader, DatasetRecord } from "./types";

/**
 * BCI Competition IV dataset 2a — 22-channel motor imagery, 250 Hz.
 * Requires an HTTPS mirror staged by the operator with files:
 *   {mirrorBase}/A0{1..9}{T|E}.edf
 */
export function bciCompetitionIV2a(mirrorBase: string): DatasetLoader {
  if (!mirrorBase) throw new Error("bciCompetitionIV2a: mirrorBase required");
  const base = mirrorBase.replace(/\/$/, "");
  return {
    name: "bci-competition-iv-2a",
    async list(): Promise<DatasetRecord[]> {
      const records: DatasetRecord[] = [];
      for (let s = 1; s <= 9; s++) {
        for (const phase of ["T", "E"] as const) {
          const file = `A0${s}${phase}.edf`;
          records.push({
            id: file,
            subject: `A0${s}`,
            session: phase === "T" ? "train" : "eval",
            task: "motor-imagery-4class",
            url: `${base}/${file}`,
            format: "edf",
          });
        }
      }
      return records;
    },
    async load(record, fetcher = fetch): Promise<EEGSignal> {
      const res = await fetcher(record.url);
      if (!res.ok) throw new Error(`bci-iv-2a: ${res.status} ${res.statusText}`);
      return parseEDF(await res.arrayBuffer());
    },
  };
}