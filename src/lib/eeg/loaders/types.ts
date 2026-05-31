import type { EEGSignal } from "../types";

export interface DatasetRecord {
  id: string;
  subject: string;
  session?: string;
  task?: string;
  url: string;
  format: "edf" | "csv" | "npy";
  sampleRate?: number;
}

export interface DatasetLoader {
  name: string;
  list(): Promise<DatasetRecord[]>;
  load(record: DatasetRecord, fetcher?: typeof fetch): Promise<EEGSignal>;
}