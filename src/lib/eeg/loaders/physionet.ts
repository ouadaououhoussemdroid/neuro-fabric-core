import { parseEDF } from "../parsers/edf";
import type { EEGSignal } from "../types";
import type { DatasetLoader, DatasetRecord } from "./types";

/**
 * PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb), v1.0.0.
 *   https://physionet.org/files/eegmmidb/1.0.0/S{subject:003d}/S{subject:003d}R{run:002d}.edf
 */
const BASE = "https://physionet.org/files/eegmmidb/1.0.0";
const pad = (n: number, w: number) => n.toString().padStart(w, "0");

export const physionet: DatasetLoader = {
  name: "physionet-eegmmidb",
  async list(): Promise<DatasetRecord[]> {
    const records: DatasetRecord[] = [];
    for (let subject = 1; subject <= 109; subject++) {
      for (let run = 1; run <= 14; run++) {
        const sId = `S${pad(subject, 3)}`;
        const file = `${sId}R${pad(run, 2)}.edf`;
        records.push({
          id: file,
          subject: sId,
          session: `run-${pad(run, 2)}`,
          task: run <= 2 ? "baseline" : "motor-imagery",
          url: `${BASE}/${sId}/${file}`,
          format: "edf",
        });
      }
    }
    return records;
  },
  async load(record, fetcher = fetch): Promise<EEGSignal> {
    const res = await fetcher(record.url);
    if (!res.ok) throw new Error(`physionet: ${res.status} ${res.statusText}`);
    return parseEDF(await res.arrayBuffer());
  },
};