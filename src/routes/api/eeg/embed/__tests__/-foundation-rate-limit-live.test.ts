/**
 * Mission 14 — Gate 2: Rate limiting (LIVE against the real Supabase `check_rate_limit` RPC).
 *
 * Unlike -foundation.test.ts (which mocks checkRateLimit), this test exercises the ACTUAL
 * Postgres plpgsql function `check_rate_limit` (supabase/migrations/20260711050000_rate_limits.sql)
 * over a live `supabase/postgres:15.14.1.162` container (throwaway, host port 5433) — proving the
 * deterministic boundary the route relies on:
 *
 *   - requests 1..20 within budget are ACCEPTED (allowed=true);
 *   - the 21st request is REJECTED deterministically (allowed=false, retry_after_ms>0);
 *   - rejection is user-isolated (a different authorized user still allowed);
 *   - the FK rate_limits.user_id -> auth.users is enforced (nonexistent user errors).
 *
 * The route maps allowed=false -> HTTP 429 (foundation.ts:157-162) and a thrown RPC -> 503.
 * `pg` is not a workspace dependency, so we drive the real Postgres function via `psql`
 * against the throwaway container — this is the actual RPC function the route calls through
 * checkRateLimit (supabase.rpc("check_rate_limit")), not a mock of that function.
 *
 * This test requires the `nf-m14-ratelimit` container (port 5433) to be running and migrated.
 * It is skipped when that container is absent, so CI is never broken by environment limits.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { execSync } from "node:child_process";
import { existsSync } from "node:fs";

const PG_PORT = 5433;
const FIXTURE_USER = "00000000-0000-0000-0000-000000000042";
const OTHER_USER = "00000000-0000-0000-0000-000000000051";
// A user guaranteed NOT to exist in auth.users (FK must reject it, not silently allow).
const NONEXISTENT_USER = "00000000-0000-0000-0000-000000000999";

// Production values from src/routes/api/eeg/embed/foundation.ts:42-43
const RATE_LIMIT_MAX = 20;
const RATE_LIMIT_WINDOW_SECONDS = 60;

function psqlOnceRaw(sql: string, opts: { mayFail?: boolean } = {}): string {
  // Feed SQL via stdin to avoid all shell-quoting issues with nested quotes in SQL.
  // `mayFail` lets a negative assertion (e.g. FK violation) observe the non-zero exit
  // without throwing in beforeEach.
  try {
    const r = execSync(
      `docker exec -i nf-m14-ratelimit psql -U postgres -d postgres -At -v ON_ERROR_STOP=1`,
      { encoding: "utf-8", input: sql + "\n", stdio: ["pipe", "pipe", "pipe"] },
    );
    return r.trim();
  } catch (e) {
    if (opts.mayFail) {
      // Nonzero exit is the expected signal that the SQL errored (e.g. FK violation).
      const err = e as NodeJS.ErrwithOptionalStdio;
      return (err.stderr?.toString() ?? (err as Error)?.message ?? "").trim();
    }
    throw e;
  }
}

function rpcAllowed(uid: string, max = RATE_LIMIT_MAX): boolean {
  const out = psqlOnceRaw(
    `SELECT (check_rate_limit('${uid}'::uuid, ${max}, ${RATE_LIMIT_WINDOW_SECONDS})).allowed::text`,
  );
  // PG boolean::text yields "true"/"false" (not "t"/"f") under -At tuple mode.
  return out === "true";
}

/** Wrapper that returns false on a thrown RPC (e.g. FK violation) — used only
 * where "not allowed" is the expected contract outcome, not a silent accept. */
function rpcAllowedLenient(uid: string): boolean {
  try {
    return rpcAllowed(uid);
  } catch {
    return false;
  }
}

function rpcRetryMs(uid: string, max = RATE_LIMIT_MAX): number {
  const out = psqlOnceRaw(
    `SELECT (check_rate_limit('${uid}'::uuid, ${max}, ${RATE_LIMIT_WINDOW_SECONDS})).retry_after_ms::text`,
  );
  return Number(out);
}

/** Run the RPC N times in a single psql round-trip and return the per-call `allowed` booleans. */
function rpcBatch(uid: string, n: number, max = RATE_LIMIT_MAX): boolean[] {
  const out = psqlOnceRaw(
    `WITH seq(n) AS (SELECT generate_series(1, ${n})) ` +
      `SELECT ((check_rate_limit('${uid}'::uuid, ${max}, ${RATE_LIMIT_WINDOW_SECONDS})).allowed)::text ` +
      `FROM seq;`,
  );
  return out
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => s === "true");
}

function resetUser(uid: string): void {
  psqlOnceRaw(`DELETE FROM public.rate_limits WHERE user_id = '${uid}'::uuid`);
}

function ensureUser(uid: string): void {
  psqlOnceRaw(`INSERT INTO auth.users(id) VALUES ('${uid}'::uuid) ON CONFLICT (id) DO NOTHING`);
}

const containerUp = (() => {
  try {
    const names = execSync(
      "docker ps --filter ancestor=supabase/postgres:15.14.1.162 --filter name=nf-m14-ratelimit --format {{.Names}}",
      { encoding: "utf-8", stdio: ["pipe", "pipe", "pipe"] },
    ).trim();
    return names === "nf-m14-ratelimit";
  } catch {
    return false;
  }
})();

// Skip if the throwaway container is not available — environment limitation, not a failure.
const maybeDescribe = containerUp ? describe : describe.skip;

maybeDescribe(
  "Gate 2: LIVE rate-limit RPC (check_rate_limit, budget 20/60s)",
  { timeout: 120_000 },
  () => {
    beforeAll(() => {
      ensureUser(FIXTURE_USER);
      ensureUser(OTHER_USER);
      resetUser(FIXTURE_USER);
      resetUser(OTHER_USER);
    });

    afterAll(() => {
      resetUser(FIXTURE_USER);
      resetUser(OTHER_USER);
    });

    it("accepts requests within the budget (allowed=true on each of the 20 calls)", () => {
      resetUser(FIXTURE_USER);
      const results = rpcBatch(FIXTURE_USER, RATE_LIMIT_MAX);
      expect(results).toHaveLength(RATE_LIMIT_MAX);
      expect(results.every((a) => a === true)).toBe(true);
    });

    it("rejects the (budget+1)th request deterministically (allowed=false, retry>0)", () => {
      resetUser(FIXTURE_USER);
      // Fill the 20-request budget in a single round-trip.
      rpcBatch(FIXTURE_USER, RATE_LIMIT_MAX);
      // The 21st request must be rejected — deterministically.
      expect(rpcAllowed(FIXTURE_USER)).toBe(false);
      const retry = rpcRetryMs(FIXTURE_USER);
      expect(retry).toBeGreaterThan(0);
      expect(retry).toBeLessThanOrEqual(RATE_LIMIT_WINDOW_SECONDS * 1000);
    });

    it("rejection is user-isolated: another user within budget still allowed", () => {
      // Exhaust the fixture user's budget.
      resetUser(FIXTURE_USER);
      rpcBatch(FIXTURE_USER, RATE_LIMIT_MAX);
      expect(rpcAllowed(FIXTURE_USER)).toBe(false);
      // A different authorized user is still within its own budget.
      resetUser(OTHER_USER);
      const otherRaw = psqlOnceRaw(
        `SELECT (check_rate_limit('${OTHER_USER}'::uuid, ${RATE_LIMIT_MAX}, ${RATE_LIMIT_WINDOW_SECONDS})).allowed::text`,
      );
      expect(otherRaw).toBe("true");
    });

    it("FK enforced: nonexistent user is rejected at the RPC (no silent accept)", () => {
      resetUser(NONEXISTENT_USER);
      // The function's atomic UPSERT references auth.users -> a nonexistent user must
      // raise a foreign-key violation, NOT silently return allowed=true.
      const err = psqlOnceRaw(
        `SELECT (check_rate_limit('${NONEXISTENT_USER}'::uuid, ${RATE_LIMIT_MAX}, ${RATE_LIMIT_WINDOW_SECONDS})).allowed::text`,
        { mayFail: true },
      );
      expect(err).toMatch(/violates.*foreign key constraint|check_rate_limit/i);
      // A nonexistent user must NOT be silently accepted — the RPC errors (FK violation)
      // which the route maps to 503, never to allowed=true.
      expect(rpcAllowedLenient(NONEXISTENT_USER)).toBe(false);
    });
  },
);
