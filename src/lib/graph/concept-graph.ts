/**
 * T-013 — Concept graph schema (subject → session → window → embedding → label).
 *
 * TypeScript types and query helpers for the property-graph schema backed by
 * Postgres `ltree` + the `embeddings` FK (migration
 * `20260711070000_concept_graph.sql`).
 *
 * The schema enables provenance queries (e.g. "which subject/session produced
 * this embedding?") without a dedicated graph database. Apache AGE is
 * reserved for the long-term scalable path once graph queries exceed ltree's
 * expressive power.
 */

/** Subject node in the concept graph. */
export interface GraphSubject {
  id: string;
  userId: string;
  subjectCode: string;
  dataset: string;
  metadata: Record<string, unknown>;
  createdAt: string;
  /** ltree path (e.g. "B01"). */
  path: string;
}

/** Session node (child of a subject). */
export interface GraphSession {
  id: string;
  subjectId: string;
  sessionCode: string;
  metadata: Record<string, unknown>;
  createdAt: string;
  /** ltree path (e.g. "B01.T1"). */
  path: string;
}

/** Window node (child of a session, optionally linked to an embedding). */
export interface GraphWindow {
  id: string;
  sessionId: string;
  embeddingId: string | null;
  windowIndex: number;
  startSample: number;
  endSample: number;
  sampleRate: number;
  label: string | null;
  metadata: Record<string, unknown>;
  createdAt: string;
  /** ltree path (e.g. "B01.T1.42"). */
  path: string;
}

/** Label node (normalised, shared across windows). */
export interface GraphLabel {
  id: string;
  dataset: string;
  label: string;
  description: string | null;
}

/** Full provenance chain for an embedding. */
export interface EmbeddingProvenance {
  subjectCode: string;
  dataset: string;
  sessionCode: string;
  windowIndex: number;
  labels: string[];
}

/**
 * Minimal Supabase client shape needed for provenance queries.
 * Matches the subset used by the rest of the codebase.
 */
export interface GraphClient {
  from: (table: string) => {
    insert: (rows: unknown) => {
      select: () => Promise<{ data: unknown[] | null; error: unknown }>;
    };
    select: (columns?: string) => {
      eq: (
        col: string,
        val: unknown,
      ) => {
        single: () => Promise<{ data: unknown | null; error: unknown }>;
        limit: (n: number) => Promise<{ data: unknown[] | null; error: unknown }>;
      };
    };
  };
  rpc: (fn: string, args: Record<string, unknown>) => Promise<{ data: unknown; error: unknown }>;
}

/**
 * Query the full provenance chain for an embedding via the
 * `get_embedding_provenance` RPC (defined in the migration).
 */
export async function getEmbeddingProvenance(
  client: GraphClient,
  embeddingId: string,
): Promise<EmbeddingProvenance | null> {
  const { data, error } = await client.rpc("get_embedding_provenance", {
    p_embedding_id: embeddingId,
  });
  if (error || !data) return null;
  
  // Handle two possible RPC response formats:
  // 1. { data: [{...}], error: null } (wrapped format)
  // 2. [{...}] (direct array format)
  let rows: unknown[] = [];
  
  if (Array.isArray(data)) {
    // Format 2: Direct array format [{...}]
    rows = data;
  } else if (typeof data === 'object' && data !== null && 'data' in data) {
    // Format 1: Wrapped format { data: [...], error: null }
    const response = data as { data?: unknown[]; error?: unknown };
    rows = response?.data ?? [];
  }
  
  if (!rows || !Array.isArray(rows) || rows.length === 0) return null;
  
  // Transform snake_case database keys to camelCase for the application
  const row = rows[0] as Record<string, unknown>;
  
  return {
    subjectCode: row.subject_code as string,
    dataset: row.dataset as string,
    sessionCode: row.session_code as string,
    windowIndex: row.window_index as number,
    labels: Array.isArray(row.labels) ? (row.labels as string[]) : [],
  };
}

/**
 * Insert a subject node. Returns the created row (or null on error).
 * The `path` is auto-populated by the DB trigger.
 */
export async function createSubject(
  client: GraphClient,
  input: Omit<GraphSubject, "id" | "createdAt" | "path">,
): Promise<GraphSubject | null> {
  const { data, error } = await client
    .from("graph_subjects")
    .insert({
      user_id: input.userId,
      subject_code: input.subjectCode,
      dataset: input.dataset,
      metadata: input.metadata,
    })
    .select();
  if (error || !data || data.length === 0) return null;
  return mapSubject(data[0] as Record<string, unknown>);
}

/**
 * Insert a session node. Returns the created row.
 */
export async function createSession(
  client: GraphClient,
  input: Omit<GraphSession, "id" | "createdAt" | "path">,
): Promise<GraphSession | null> {
  const { data, error } = await client
    .from("graph_sessions")
    .insert({
      subject_id: input.subjectId,
      session_code: input.sessionCode,
      metadata: input.metadata,
    })
    .select();
  if (error || !data || data.length === 0) return null;
  return mapSession(data[0] as Record<string, unknown>);
}

/**
 * Insert a window node. Returns the created row.
 */
export async function createWindow(
  client: GraphClient,
  input: Omit<GraphWindow, "id" | "createdAt" | "path">,
): Promise<GraphWindow | null> {
  const { data, error } = await client
    .from("graph_windows")
    .insert({
      session_id: input.sessionId,
      embedding_id: input.embeddingId,
      window_index: input.windowIndex,
      start_sample: input.startSample,
      end_sample: input.endSample,
      sample_rate: input.sampleRate,
      label: input.label,
      metadata: input.metadata,
    })
    .select();
  if (error || !data || data.length === 0) return null;
  return mapWindow(data[0] as Record<string, unknown>);
}

// --- Row mappers (snake_case → camelCase) ---

function mapSubject(row: Record<string, unknown>): GraphSubject {
  return {
    id: row.id as string,
    userId: row.user_id as string,
    subjectCode: row.subject_code as string,
    dataset: row.dataset as string,
    metadata: (row.metadata as Record<string, unknown>) ?? {},
    createdAt: row.created_at as string,
    path: String(row.path ?? ""),
  };
}

function mapSession(row: Record<string, unknown>): GraphSession {
  return {
    id: row.id as string,
    subjectId: row.subject_id as string,
    sessionCode: row.session_code as string,
    metadata: (row.metadata as Record<string, unknown>) ?? {},
    createdAt: row.created_at as string,
    path: String(row.path ?? ""),
  };
}

function mapWindow(row: Record<string, unknown>): GraphWindow {
  return {
    id: row.id as string,
    sessionId: row.session_id as string,
    embeddingId: (row.embedding_id as string | null) ?? null,
    windowIndex: row.window_index as number,
    startSample: row.start_sample as number,
    endSample: row.end_sample as number,
    sampleRate: row.sample_rate as number,
    label: (row.label as string | null) ?? null,
    metadata: (row.metadata as Record<string, unknown>) ?? {},
    createdAt: row.created_at as string,
    path: String(row.path ?? ""),
  };
}
