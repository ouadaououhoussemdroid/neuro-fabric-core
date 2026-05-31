import { parseEDF } from "../parsers/edf";
import type { EEGSignal } from "../types";
import type { DatasetLoader, DatasetRecord } from "./types";

/**
 * TUH EEG Corpus loader — ARCHITECTURE ONLY.
 * Access to TUH requires credentialed rsync/SFTP at isip.piconepress.com.
 * The operator must stage an HTTPS mirror and supply a prebuilt index.
 * Without both, list() returns [] (no mock records emitted).
 */
export interface TuhIndexEntry {
  subject: string;
  session: string;
  file: string;
}

export function tuhEeg(mirrorBase?: string, index: TuhIndexEntry[] = []): DatasetLoader {
  const base = (mirrorBase ?? "").replace(/\/$/, "");
  return {
    name: "tuh-eeg",
    async list(): Promise<DatasetRecord[]> {
      if (!base || index.length === 0) return [];
      return index.map((e) => ({
        id: `${e.subject}/${e.session}/${e.file}`,
        subject: e.subject,
        session: e.session,
        url: `${base}/${e.file}`,
        format: "edf",
      }));
    },
    async load(record, fetcher = fetch): Promise<EEGSignal> {
      if (!base) throw new Error("tuh: mirrorBase not configured");
      const res = await fetcher(record.url);
      if (!res.ok) throw new Error(`tuh: ${res.status} ${res.statusText}`);
      return parseEDF(await res.arrayBuffer());
    },
  };
}