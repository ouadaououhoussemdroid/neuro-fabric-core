import { parseEDF } from "../parsers/edf";
import type { EEGSignal } from "../types";
import type { DatasetLoader, DatasetRecord } from "./types";

/**
 * Sleep-EDF Database (v1.0.0) loader.
 *
 * The Sleep-EDF database contains polysomnographic recordings from the
 * Sleep Heart Birth Cohort and other cohorts. The SC (Sleep Cassette) subset
 * provides Fpz-Cz + EOG montages sampled at 100 Hz, with paired hypnograms.
 *
 * Reference: https://physionet.org/files/sleep-edf/1.0.0/
 *
 * URL layout (nightly 1.0.0 version):
 *   SC/subjectXX/SC_subjectXX.edf          — PSG recording
 *   SC/subjectXX/SC_subjectXX_hypnogram.json — sleep stage labels
 *
 * This loader indexes SC subjects 000–099 (100 subjects). Each subject has
 * one night of PSG data; subject 000 has two nights.
 */
const BASE = "https://physionet.org/files/sleep-edf/1.0.0";
const PAD = (n: number, w: number) => n.toString().padStart(w, "0");

export const sleepEDF: DatasetLoader = {
  name: "sleep-edf-sc",
  async list(): Promise<DatasetRecord[]> {
    const records: DatasetRecord[] = [];
    const MAX_SUBJECT = 99;
    for (let subject = 0; subject <= MAX_SUBJECT; subject++) {
      const sId = `S${PAD(subject, 4)}`;
      const night = subject === 0 ? 2 : 1;
      for (let n = 1; n <= night; n++) {
        const file = `${sId}-n${n}.edf`;
        const url = `${BASE}/SC/${sId}/${file}`;
        records.push({
          id: `${sId}-n${n}`,
          subject: sId,
          session: `night-${n}`,
          task: "sleep-staging",
          url,
          format: "edf",
          sampleRate: 100,
        });
      }
    }
    return records;
  },
  async load(record, fetcher = fetch): Promise<EEGSignal> {
    const res = await fetcher(record.url);
    if (!res.ok) {
      throw new Error(`sleep-edf: ${res.status} ${res.statusText} (${record.url})`);
    }
    return parseEDF(await res.arrayBuffer());
  },
};
