import { describe, it, expect, afterEach } from "vitest";
import { setRolloutStage, isEEGConformerEnabledForUser } from "../rollout";

describe("EEGConformer cohort routing", () => {
  afterEach(() => {
    setRolloutStage("off");
  });

  it("returns false for all users when stage is off", () => {
    setRolloutStage("off");
    expect(isEEGConformerEnabledForUser("user-1")).toBe(false);
    expect(isEEGConformerEnabledForUser("user-2")).toBe(false);
    expect(isEEGConformerEnabledForUser(undefined)).toBe(false);
  });

  it("returns true for all users when stage is ga", () => {
    setRolloutStage("ga");
    expect(isEEGConformerEnabledForUser("user-1")).toBe(true);
    expect(isEEGConformerEnabledForUser("user-2")).toBe(true);
    expect(isEEGConformerEnabledForUser(undefined)).toBe(true);
  });

  it("returns false when no userId is provided for canary", () => {
    setRolloutStage("canary");
    expect(isEEGConformerEnabledForUser(undefined)).toBe(false);
  });

  it("returns false when no userId is provided for beta", () => {
    setRolloutStage("beta");
    expect(isEEGConformerEnabledForUser(undefined)).toBe(false);
  });

  it("canary enables approximately 5% of users", () => {
    setRolloutStage("canary");
    let count = 0;
    for (let i = 0; i < 1000; i++) {
      if (isEEGConformerEnabledForUser(`user-${i}`)) count++;
    }
    expect(count).toBeGreaterThan(30);
    expect(count).toBeLessThan(80);
  });

  it("beta enables approximately 50% of users", () => {
    setRolloutStage("beta");
    let count = 0;
    for (let i = 0; i < 1000; i++) {
      if (isEEGConformerEnabledForUser(`user-${i}`)) count++;
    }
    expect(count).toBeGreaterThan(400);
    expect(count).toBeLessThan(600);
  });

  it("is deterministic for the same userId", () => {
    setRolloutStage("canary");
    const first = isEEGConformerEnabledForUser("user-123");
    const second = isEEGConformerEnabledForUser("user-123");
    expect(first).toBe(second);
  });
});
