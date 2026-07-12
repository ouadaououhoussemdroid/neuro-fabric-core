/**
 * T-019 — Dataset manifest types.
 *
 * TypeScript types for the `datasets` table (migration
 * `20260711080000_datasets_manifest.sql`). The table tracks reproducibility
 * metadata (name, license, sha256, source URL, preprocessing hash) for every
 * dataset the platform trains on or evaluates against.
 */

/** A dataset entry in the manifest. */
export interface DatasetManifestEntry {
  id: string;
  userId: string | null;
  name: string;
  license: string;
  rawSha256: string | null;
  sourceUrl: string | null;
  nSubjects: number | null;
  nChannels: number | null;
  sampleRate: number | null;
  nClasses: number | null;
  preprocessingSha256: string | null;
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

/** Known public datasets pre-seeded in the manifest (for reference). */
export const KNOWN_DATASETS = [
  {
    name: "BCI-IV-2a",
    license: "BSD-3-Clause",
    sourceUrl: "http://bnci-horizon-2020.eu/database/data-sets-001-004",
    nSubjects: 9,
    nChannels: 22,
    sampleRate: 250,
    nClasses: 4,
    metadata: { paradigm: "motor_imagery", classes: ["left_hand", "right_hand", "feet", "tongue"] },
  },
  {
    name: "BCI-IV-2b",
    license: "BSD-3-Clause",
    sourceUrl: "http://bnci-horizon-2020.eu/database/data-sets-001-004",
    nSubjects: 9,
    nChannels: 3,
    sampleRate: 250,
    nClasses: 2,
    metadata: { paradigm: "motor_imagery", classes: ["left_hand", "right_hand"] },
  },
  {
    name: "PhysioNetMI",
    license: "CC-BY-4.0",
    sourceUrl: "https://physionet.org/content/eegmmidb/1.0.0/",
    nSubjects: 109,
    nChannels: 64,
    sampleRate: 160,
    nClasses: 4,
    metadata: { paradigm: "motor_imagery" },
  },
] as const;

/** Minimal Supabase client shape for dataset queries. */
export interface DatasetClient {
  from: (table: string) => {
    insert: (rows: unknown) => {
      select: () => Promise<{ data: unknown[] | null; error: unknown }>;
    };
    select: (columns?: string) => {
      order: (col: string, opts?: unknown) => Promise<{ data: unknown[] | null; error: unknown }>;
    };
    delete: () => { eq: (col: string, val: unknown) => Promise<{ error: unknown }> };
  };
}

/** Map a DB row (snake_case) to a DatasetManifestEntry (camelCase). */
export function mapDatasetRow(row: Record<string, unknown>): DatasetManifestEntry {
  return {
    id: row.id as string,
    userId: (row.user_id as string | null) ?? null,
    name: row.name as string,
    license: row.license as string,
    rawSha256: (row.raw_sha256 as string | null) ?? null,
    sourceUrl: (row.source_url as string | null) ?? null,
    nSubjects: (row.n_subjects as number | null) ?? null,
    nChannels: (row.n_channels as number | null) ?? null,
    sampleRate: (row.sample_rate as number | null) ?? null,
    nClasses: (row.n_classes as number | null) ?? null,
    preprocessingSha256: (row.preprocessing_sha256 as string | null) ?? null,
    metadata: (row.metadata as Record<string, unknown>) ?? {},
    createdAt: row.created_at as string,
    updatedAt: row.updated_at as string,
  };
}

/** List all datasets in the manifest, ordered by name. */
export async function listDatasets(client: DatasetClient): Promise<DatasetManifestEntry[]> {
  const { data, error } = await client
    .from("datasets")
    .select("*")
    .order("name", { ascending: true });
  if (error || !data) return [];
  return (data as Record<string, unknown>[]).map(mapDatasetRow);
}

/** Insert a dataset into the manifest. Returns the created entry or null. */
export async function insertDataset(
  client: DatasetClient,
  entry: Omit<DatasetManifestEntry, "id" | "createdAt" | "updatedAt" | "userId"> & {
    userId?: string;
  },
): Promise<DatasetManifestEntry | null> {
  const { data, error } = await client
    .from("datasets")
    .insert({
      user_id: entry.userId ?? null,
      name: entry.name,
      license: entry.license,
      raw_sha256: entry.rawSha256,
      source_url: entry.sourceUrl,
      n_subjects: entry.nSubjects,
      n_channels: entry.nChannels,
      sample_rate: entry.sampleRate,
      n_classes: entry.nClasses,
      preprocessing_sha256: entry.preprocessingSha256,
      metadata: entry.metadata,
    })
    .select();
  if (error || !data || data.length === 0) return null;
  return mapDatasetRow(data[0] as Record<string, unknown>);
}
