import { z } from "zod";

// Server-only. Do not import from client code — see client.server.ts's
// own note on the .server.ts naming convention.
//
// IMPORTANT: on Cloudflare Workers, process.env only binds at REQUEST
// time (see config.server.ts) — a module-scope call to requireServerEnv
// would always see an empty env and fail. Call this lazily, inside a
// function/handler, exactly like the Proxy-wrapped clients already do.

const ENV_SCHEMAS = {
  SUPABASE_URL: z.string().url(),
  SUPABASE_PUBLISHABLE_KEY: z.string().min(1),
  SUPABASE_SERVICE_ROLE_KEY: z.string().min(1),
  CRON_SECRET: z.string().min(1),
  AI_EEGCONFORMER_ENABLED: z.enum(["off", "canary", "beta", "ga"]),
} as const;

type EnvKey = keyof typeof ENV_SCHEMAS;

/**
 * Validates the given server env vars against their schema, returning a
 * typed record on success. Throws a single, consistently-worded error
 * (replacing the ad hoc per-module checks previously duplicated across
 * client.server.ts and request-auth.ts) listing everything missing or
 * invalid at once, rather than failing on the first bad key.
 */
export function requireServerEnv<K extends EnvKey>(keys: readonly K[]): Record<K, string> {
  const out = {} as Record<K, string>;
  const missing: string[] = [];
  const invalid: string[] = [];

  for (const key of keys) {
    const raw = process.env[key];
    if (raw === undefined || raw === "") {
      missing.push(key);
      continue;
    }
    const result = ENV_SCHEMAS[key].safeParse(raw);
    if (!result.success) {
      invalid.push(key);
      continue;
    }
    out[key] = result.data;
  }

  if (missing.length > 0 || invalid.length > 0) {
    const parts = [
      missing.length > 0 ? `missing: ${missing.join(", ")}` : null,
      invalid.length > 0 ? `invalid: ${invalid.join(", ")}` : null,
    ]
      .filter(Boolean)
      .join("; ");
    throw new Error(`Server misconfigured (${parts}). Connect Supabase in Lovable Cloud.`);
  }

  return out;
}

/**
 * EEGConformer rollout stage, read from the AI_EEGCONFORMER_ENABLED env var.
 * Defaults to "off" when unset or invalid — the feature is opt-in.
 */
export type EEGConformerRolloutStage = "off" | "canary" | "beta" | "ga";

/**
 * Read the current EEGConformer rollout stage from the environment.
 * Reuses the ENV_SCHEMAS validation so an invalid value is treated the
 * same as "off" rather than crashing the request.
 */
export function getEEGConformerRolloutStage(): EEGConformerRolloutStage {
  const raw = process.env.AI_EEGCONFORMER_ENABLED;
  if (raw === undefined || raw === "") return "off";
  const result = ENV_SCHEMAS.AI_EEGCONFORMER_ENABLED.safeParse(raw);
  return result.success ? result.data : "off";
}

/**
 * Whether the EEGConformer model should be active for the current request.
 * True for canary / beta / ga; false for off.
 */
export function isEEGConformerEnabled(): boolean {
  return getEEGConformerRolloutStage() !== "off";
}
