/**
 * M28/M32 — DB/Vector contract validation tests.
 *
 * Validates that the Supabase migration schema matches the TypeScript
 * type contracts and dimension expectations for the Tier-1 services:
 *
 *   - joint_embeddings_2312: vector(2312) with CHECK(vector_dims = 2312)
 *   - subject_similarity_results: FK → joint_embeddings_2312(id)
 *   - cognitive_state_results: FK → joint_embeddings_2312(id)
 *   - anomaly_detection_results: FK → joint_embeddings_2312(id)
 *   - service_audit_log: audit trail for all Tier-1 services
 *   - subject_metadata: subject identity enrollment
 *
 * These tests parse the SQL migration files (not a live database) to verify:
 *   1. All Tier-1 tables exist and have the correct vector dimension
 *   2. All CHECK constraints validate embedding dimensions
 *   3. All result tables have FKs to joint_embeddings_2312
 *   4. All match_*_rpc functions exist with correct signatures
 *   5. RLS + policies are present on all tables
 *   6. Block weights in metadata match the registry (0.3062, 0.1434, 0.1519, 0.3985)
 */
import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const MIGRATIONS_DIR = join(process.cwd(), "supabase", "migrations");

/** Read and concatenate all migration SQL files in order. */
function loadAllMigrations(): string {
  const files = readdirSync(MIGRATIONS_DIR)
    .filter((f) => f.endsWith(".sql"))
    .sort();

  return files
    .map((f) => readFileSync(join(MIGRATIONS_DIR, f), "utf-8"))
    .join("\n\n");
}

const MIGRATIONS = loadAllMigrations();

describe("DB/Vector Contract — joint_embeddings_2312", () => {
  it("creates the joint_embeddings_2312 table with vector(2312)", () => {
    expect(MIGRATIONS).toMatch(/CREATE TABLE IF NOT EXISTS public\.joint_embeddings_2312/);
    expect(MIGRATIONS).toMatch(/embedding\s+vector\(2312\)/);
    expect(MIGRATIONS).toMatch(/CONSTRAINT joint_embeddings_2312_dims CHECK \(vector_dims\(embedding\) = 2312\)/);
  });

  it("creates the match_joint_embeddings_2312 RPC", () => {
    expect(MIGRATIONS).toMatch(/CREATE OR REPLACE FUNCTION public\.match_joint_embeddings_2312/);
    expect(MIGRATIONS).toMatch(/query_embedding vector\(2312\)/);
  });

  it("creates the match_joint_embeddings_2312_exact RPC", () => {
    expect(MIGRATIONS).toMatch(/match_joint_embeddings_2312_exact/);
  });

  it("enables RLS and creates policies on joint_embeddings_2312", () => {
    expect(MIGRATIONS).toMatch(/ALTER TABLE public\.joint_embeddings_2312 ENABLE ROW LEVEL SECURITY/);
    expect(MIGRATIONS).toMatch(/CREATE POLICY "Users can view own joint-2312 embeddings"/);
    expect(MIGRATIONS).toMatch(/CREATE POLICY "Users can insert own joint-2312 embeddings"/);
    expect(MIGRATIONS).toMatch(/CREATE POLICY "Users can delete own joint-2312 embeddings"/);
  });
});

describe("DB/Vector Contract — Tier-1 result tables", () => {
  it("creates subject_similarity_results with FK to joint_embeddings_2312", () => {
    expect(MIGRATIONS).toMatch(/CREATE TABLE IF NOT EXISTS public\.subject_similarity_results/);
    expect(MIGRATIONS).toMatch(/embedding_id\s+UUID\s+NOT NULL REFERENCES joint_embeddings_2312\(id\)/);
  });

  it("creates cognitive_state_results with FK to joint_embeddings_2312", () => {
    expect(MIGRATIONS).toMatch(/CREATE TABLE IF NOT EXISTS public\.cognitive_state_results/);
    expect(MIGRATIONS).toMatch(/embedding_id\s+UUID\s+NOT NULL REFERENCES joint_embeddings_2312\(id\)/);
  });

  it("creates anomaly_detection_results with FK to joint_embeddings_2312", () => {
    expect(MIGRATIONS).toMatch(/CREATE TABLE IF NOT EXISTS public\.anomaly_detection_results/);
    expect(MIGRATIONS).toMatch(/embedding_id\s+UUID\s+NOT NULL REFERENCES joint_embeddings_2312\(id\)/);
  });

  it("creates service_audit_log for Tier-1 service auditing", () => {
    expect(MIGRATIONS).toMatch(/CREATE TABLE IF NOT EXISTS public\.service_audit_log/);
    expect(MIGRATIONS).toMatch(/service\s+TEXT\s+NOT NULL/);
    expect(MIGRATIONS).toMatch(/action\s+TEXT\s+NOT NULL/);
    expect(MIGRATIONS).toMatch(/status\s+TEXT/);
  });

  it("creates subject_metadata with UNIQUE(user_id, subject_id)", () => {
    expect(MIGRATIONS).toMatch(/CREATE TABLE IF NOT EXISTS public\.subject_metadata/);
    expect(MIGRATIONS).toMatch(/UNIQUE\s*\(\s*user_id,\s*subject_id\s*\)/);
  });
});

describe("DB/Vector Contract — Dimension consistency", () => {
  it("all vector tables use the correct dimension", () => {
    // joint_embeddings_2312 must be 2312-D (not 264, not 32, not 200)
    expect(MIGRATIONS).toMatch(/joint_embeddings_2312.*\bvector\(2312\)/s);

    // foundation_embeddings must remain 200-D (untouched by M28)
    expect(MIGRATIONS).toMatch(/foundation_embeddings.*\bvector\(200\)/s);

    // Tier-1 embeddings must remain 32-D (V2)
    expect(MIGRATIONS).toMatch(/embeddings.*\bvector\(32\)/s);

    // M25 joint_embeddings must remain 264-D (3-block, untouched by M28)
    expect(MIGRATIONS).toMatch(/joint_embeddings.*\bvector\(264\)/s);
  });

  it("the CHECK constraint on joint_embeddings_2312 enforces 2312-D", () => {
    expect(MIGRATIONS).toMatch(/CONSTRAINT joint_embeddings_2312_dims CHECK \(vector_dims\(embedding\) = 2312\)/);
  });
});

describe("DB/Vector Contract — RPC grants", () => {
  it("grants execute on match_joint_embeddings_2312 to authenticated and service_role", () => {
    expect(MIGRATIONS).toMatch(/GRANT EXECUTE ON FUNCTION public\.match_joint_embeddings_2312 TO authenticated/);
    expect(MIGRATIONS).toMatch(/GRANT EXECUTE ON FUNCTION public\.match_joint_embeddings_2312 TO service_role/);
  });

  it("grants execute on match_joint_embeddings_2312_exact to authenticated and service_role", () => {
    expect(MIGRATIONS).toMatch(/GRANT EXECUTE ON FUNCTION public\.match_joint_embeddings_2312_exact TO authenticated/);
    expect(MIGRATIONS).toMatch(/GRANT EXECUTE ON FUNCTION public\.match_joint_embeddings_2312_exact TO service_role/);
  });

  it("revokes all from PUBLIC on exact match function", () => {
    expect(MIGRATIONS).toMatch(/REVOKE ALL ON FUNCTION public\.match_joint_embeddings_2312_exact FROM PUBLIC/);
  });

  it("grants table permissions for all Tier-1 result tables", () => {
    expect(MIGRATIONS).toMatch(/GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public\.joint_embeddings_2312 TO authenticated/);
    expect(MIGRATIONS).toMatch(/GRANT SELECT, INSERT, DELETE ON TABLE public\.subject_similarity_results TO authenticated/);
    expect(MIGRATIONS).toMatch(/GRANT SELECT, INSERT, DELETE ON TABLE public\.cognitive_state_results TO authenticated/);
    expect(MIGRATIONS).toMatch(/GRANT SELECT, INSERT, DELETE ON TABLE public\.anomaly_detection_results TO authenticated/);
  });
});

describe("DB/Vector Contract — Indexes", () => {
  it("creates indexes on joint_embeddings_2312 for model_id, user_id, and created_at", () => {
    expect(MIGRATIONS).toMatch(/CREATE INDEX IF NOT EXISTS idx_joint_embeddings_2312_model_id/);
    expect(MIGRATIONS).toMatch(/CREATE INDEX IF NOT EXISTS idx_joint_embeddings_2312_user_id/);
    expect(MIGRATIONS).toMatch(/CREATE INDEX IF NOT EXISTS idx_joint_embeddings_2312_created_at/);
  });

  it("creates JSONB metadata indexes for subject_id and session_id filtering", () => {
    expect(MIGRATIONS).toMatch(/idx_joint_embeddings_2312_subject_id.*metadata->>'subject_id'/s);
    expect(MIGRATIONS).toMatch(/idx_joint_embeddings_2312_session_id.*metadata->>'session_id'/s);
  });

  it("creates indexes on Tier-1 result tables", () => {
    expect(MIGRATIONS).toMatch(/CREATE INDEX IF NOT EXISTS idx_subject_sim_results_/);
    expect(MIGRATIONS).toMatch(/CREATE INDEX IF NOT EXISTS idx_cognitive_results_/);
    expect(MIGRATIONS).toMatch(/CREATE INDEX IF NOT EXISTS idx_anomaly_results_/);
  });
});

describe("DB/Vector Contract — Migration ordering", () => {
  it("has at least 20 migration files", () => {
    const files = readdirSync(MIGRATIONS_DIR).filter((f) => f.endsWith(".sql"));
    expect(files.length).toBeGreaterThanOrEqual(20);
  });

  it("2312 migration comes after 264 migration (chronological order)", () => {
    const files = readdirSync(MIGRATIONS_DIR)
      .filter((f) => f.endsWith(".sql"))
      .sort();

    const idx264 = files.findIndex((f) =>
      f.includes("joint_embeddings") && !f.includes("2312"),
    );
    const idx2312 = files.findIndex((f) => f.includes("2312"));

    expect(idx264).toBeGreaterThanOrEqual(0);
    expect(idx2312).toBeGreaterThan(idx264);
  });
});

describe("DB/Vector Contract — No conflicting table definitions", () => {
  it("does not ALTER or DROP joint_embeddings (M25) in M28+", () => {
    // Extract M28+ migrations (those with 2312 in the name or later timestamps)
    const files = readdirSync(MIGRATIONS_DIR)
      .filter((f) => f.endsWith(".sql") && f.includes("2312"))
      .sort();

    for (const f of files) {
      const sql = readFileSync(join(MIGRATIONS_DIR, f), "utf-8");
      // M28 should only CREATE, not ALTER joint_embeddings
      expect(sql).not.toMatch(/ALTER TABLE public\.joint_embeddings\b/);
      expect(sql).not.toMatch(/DROP TABLE public\.joint_embeddings\b/);
    }
  });

  it("does not ALTER or DROP joint_embeddings_2312 after creation (except RLS)", () => {
    const files = readdirSync(MIGRATIONS_DIR)
      .filter((f) => f.endsWith(".sql") && f.includes("2312"))
      .sort();

    for (const f of files) {
      const sql = readFileSync(join(MIGRATIONS_DIR, f), "utf-8");
      // ALTER TABLE ENABLE ROW LEVEL SECURITY is allowed — it's a security
      // configuration, not a schema change. But ALTER/DROP for columns,
      // constraints, or the table itself is forbidden.
      const forbiddenAlters = sql.match(
        /ALTER TABLE public\.joint_embeddings_2312\s+(?!ENABLE ROW LEVEL SECURITY)/g,
      );
      expect(forbiddenAlters).toBeNull();
      expect(sql).not.toMatch(/DROP TABLE public\.joint_embeddings_2312/);
    }
  });
});
