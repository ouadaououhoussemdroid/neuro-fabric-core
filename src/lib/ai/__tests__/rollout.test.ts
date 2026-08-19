import { describe, it, expect, afterEach } from "vitest";
import { applyEEGConformerRollout } from "../rollout.server";
import { hasModel, registerBraindecodeEEGConformer } from "../models/registry";

const ORIGINAL_ENV = { ...process.env };
const EEGCONFORMER_ID = "braindecode-eegconformer-prod-v2";

describe("applyEEGConformerRollout", () => {
  afterEach(() => {
    process.env = { ...ORIGINAL_ENV };
    // Restore V2 registration so other tests see the default state.
    if (!hasModel(EEGCONFORMER_ID)) {
      registerBraindecodeEEGConformer({
        id: "braindecode-eegconformer-prod-v2",
        artifact: "/models/eegconformer_finetuned.onnx",
        enableVerification: true,
      });
    }
  });

  it("unregisters EEGConformer when stage is off", () => {
    process.env.AI_EEGCONFORMER_ENABLED = "off";
    applyEEGConformerRollout();
    expect(hasModel(EEGCONFORMER_ID)).toBe(false);
  });

  it("keeps EEGConformer registered when stage is canary", () => {
    process.env.AI_EEGCONFORMER_ENABLED = "canary";
    applyEEGConformerRollout();
    expect(hasModel(EEGCONFORMER_ID)).toBe(true);
  });

  it("re-registers EEGConformer after a rollback to off then ga", () => {
    process.env.AI_EEGCONFORMER_ENABLED = "off";
    applyEEGConformerRollout();
    expect(hasModel(EEGCONFORMER_ID)).toBe(false);

    process.env.AI_EEGCONFORMER_ENABLED = "ga";
    applyEEGConformerRollout();
    expect(hasModel(EEGCONFORMER_ID)).toBe(true);
  });

  it("defaults to off (unregister) when env var is missing", () => {
    delete process.env.AI_EEGCONFORMER_ENABLED;
    applyEEGConformerRollout();
    expect(hasModel(EEGCONFORMER_ID)).toBe(false);
  });
});
