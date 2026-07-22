/**
 * EEGConformer rollout cohort logic.
 *
 * The rollout stage is set per-request by rollout.server.ts (called from
 * src/start.ts requestMiddleware) and read by embedEEG() to decide
 * whether a given user should receive the EEGConformer model or fall
 * back to PCA.
 */
import type { EEGConformerRolloutStage } from "../env.server";

let currentStage: EEGConformerRolloutStage = "off";

/** Set the current rollout stage (called per-request from the server). */
export function setRolloutStage(stage: EEGConformerRolloutStage): void {
  currentStage = stage;
}

/** Percentage of users that should receive EEGConformer at each stage. */
const ROLLOUT_PERCENTAGE: Record<EEGConformerRolloutStage, number> = {
  off: 0,
  canary: 5,
  beta: 50,
  ga: 100,
};

/** Simple djb2 hash of a string, mapped to 0–99. */
function hashUserId(userId: string): number {
  let hash = 0;
  for (let i = 0; i < userId.length; i++) {
    hash = (hash * 31 + userId.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % 100;
}

/**
 * Whether the given user should receive EEGConformer at the current
 * rollout stage. Returns false when the stage is off, when no userId
 * is provided for canary/beta, or when the user is outside the cohort.
 */
export function isEEGConformerEnabledForUser(userId?: string): boolean {
  const percentage = ROLLOUT_PERCENTAGE[currentStage] ?? 0;
  if (percentage === 0) return false;
  if (percentage === 100) return true;
  if (!userId) return false;
  return hashUserId(userId) < percentage;
}
