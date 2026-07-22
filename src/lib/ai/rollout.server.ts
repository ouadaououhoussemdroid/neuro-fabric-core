/**
 * EEGConformer rollout bootstrap.
 *
 * Reads the AI_EEGCONFORMER_ENABLED stage from the server env and
 * registers/unregisters the production EEGConformer model accordingly.
 * Called per-request from src/start.ts so Cloudflare Workers — where
 * process.env binds at request time — see the correct value.
 */
import { getEEGConformerRolloutStage } from "../env.server";
import { setRolloutStage } from "./rollout";
import {
  registerBraindecodeEEGConformer,
  unregisterModel,
  hasModel,
} from "./models/registry";

const EEGCONFORMER_ID = "braindecode-eegconformer-prod";

/**
 * Apply the current rollout stage to the model registry and cohort gate.
 * - off    → unregister EEGConformer (embed falls back to PCA)
 * - canary/beta/ga → ensure EEGConformer is registered; per-user cohort
 *   routing is handled by isEEGConformerEnabledForUser() in embedEEG().
 */
export function applyEEGConformerRollout(): void {
  const stage = getEEGConformerRolloutStage();
  setRolloutStage(stage);
  if (stage === "off") {
    if (hasModel(EEGCONFORMER_ID)) unregisterModel(EEGCONFORMER_ID);
  } else {
    if (!hasModel(EEGCONFORMER_ID)) {
      registerBraindecodeEEGConformer({ artifact: "/models/eegconformer.onnx" });
    }
  }
}
