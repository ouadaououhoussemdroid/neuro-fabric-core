import { describe, it, expect } from "vitest";
import {
  computeAdaptiveStim,
  computeStimFromCognitiveState,
  detectArtifacts,
  checkSessionTimeout,
  logSafetyEvent,
  getNeuroStimDiagnostics,
  resetNeuroStim,
  MAX_CURRENT_MA,
  DEFAULT_CURRENT_MA,
  SESSION_TIMEOUT_MS,
  MAX_IMPEDANCE_KOHM,
  ARTIFACT_WINDOW_SAMPLES,
  APPROVED_DEVICE_VID,
  APPROVED_DEVICE_PID,
  STIM_MODES,
  STIM_WAVEFORMS,
  type NeuralBiomarker,
  type StimParams,
  type StimSession,
  type ArtifactDetection,
  type StimDecision,
} from "../neurostimulator.browser";

describe("M54 — Adaptive Neurostimulation Protocol", () => {
  describe("Constants & Safety Limits", () => {
    it("should enforce MAX_CURRENT_MA ≤ 2.0 (FDA safe limit)", () => {
      expect(MAX_CURRENT_MA).toBe(2.0);
    });

    it("should have 30-minute session timeout", () => {
      expect(SESSION_TIMEOUT_MS).toBe(30 * 60 * 1000);
    });

    it("should have impedance limit ≤ 50kΩ", () => {
      expect(MAX_IMPEDANCE_KOHM).toBe(50.0);
    });

    it("should include FDA-approved device vendors", () => {
      expect(APPROVED_DEVICE_VID.soterix).toBe(0x16c0);
      expect(APPROVED_DEVICE_VID.openbci).toBe(0x2341);
      expect(APPROVED_DEVICE_VID.halo).toBe(0x2341);
    });

    it("should include FDA-approved product IDs", () => {
      expect(APPROVED_DEVICE_PID.soterix).toBe(0x04b4);
      expect(APPROVED_DEVICE_PID.openbci).toBe(0x0037);
    });

    it("should support standard stim modes", () => {
      expect(STIM_MODES).toContain("tDCS");
      expect(STIM_MODES).toContain("tACS");
      expect(STIM_MODES).toContain("tRNS");
    });

    it("should support standard waveforms", () => {
      expect(STIM_WAVEFORMS).toContain("dc");
      expect(STIM_WAVEFORMS).toContain("ac");
      expect(STIM_WAVEFORMS).toContain("noise");
    });
  });

  describe("computeAdaptiveStim", () => {
    it("should recommend tDCS for high fatigue", () => {
      const biomarker: NeuralBiomarker = {
        workload: 0.5, fatigue: 0.8, attention: 0.6,
        thetaBetaRatio: 1.0, alphaAsymmetry: 0.0, timestamp: Date.now(),
      };
      const decision = computeAdaptiveStim(biomarker);
      expect(decision.shouldStimulate).toBe(true);
      expect(decision.params.mode).toBe("tDCS");
      expect(decision.params.current).toBe(1.5);
    });

    it("should recommend theta tACS for low attention", () => {
      const biomarker: NeuralBiomarker = {
        workload: 0.5, fatigue: 0.3, attention: 0.2,
        thetaBetaRatio: 1.0, alphaAsymmetry: 0.0, timestamp: Date.now(),
      };
      const decision = computeAdaptiveStim(biomarker);
      expect(decision.shouldStimulate).toBe(true);
      expect(decision.params.mode).toBe("tACS");
      expect(decision.params.frequency).toBe(6.0);
    });

    it("should recommend tRNS for low workload", () => {
      const biomarker: NeuralBiomarker = {
        workload: 0.2, fatigue: 0.3, attention: 0.6,
        thetaBetaRatio: 1.0, alphaAsymmetry: 0.0, timestamp: Date.now(),
      };
      const decision = computeAdaptiveStim(biomarker);
      expect(decision.shouldStimulate).toBe(true);
      expect(decision.params.mode).toBe("tRNS");
    });

    it("should recommend alpha tACS for high theta/beta ratio", () => {
      const biomarker: NeuralBiomarker = {
        workload: 0.5, fatigue: 0.3, attention: 0.6,
        thetaBetaRatio: 2.5, alphaAsymmetry: 0.0, timestamp: Date.now(),
      };
      const decision = computeAdaptiveStim(biomarker);
      expect(decision.shouldStimulate).toBe(true);
      expect(decision.params.mode).toBe("tACS");
      expect(decision.params.frequency).toBe(10.0);
    });

    it("should not stimulate when biomarkers are normal", () => {
      const biomarker: NeuralBiomarker = {
        workload: 0.7, fatigue: 0.3, attention: 0.7,
        thetaBetaRatio: 1.0, alphaAsymmetry: 0.0, timestamp: Date.now(),
      };
      const decision = computeAdaptiveStim(biomarker);
      expect(decision.shouldStimulate).toBe(false);
    });

    it("should provide confidence score", () => {
      const biomarker: NeuralBiomarker = {
        workload: 0.2, fatigue: 0.8, attention: 0.2,
        thetaBetaRatio: 3.0, alphaAsymmetry: 0.0, timestamp: Date.now(),
      };
      const decision = computeAdaptiveStim(biomarker);
      expect(decision.confidence).toBeGreaterThan(0);
      expect(decision.confidence).toBeLessThanOrEqual(1);
    });

    it("should respect max current limit", () => {
      const biomarker: NeuralBiomarker = {
        workload: 0.2, fatigue: 0.8, attention: 0.2,
        thetaBetaRatio: 3.0, alphaAsymmetry: 0.0, timestamp: Date.now(),
      };
      const decision = computeAdaptiveStim(biomarker);
      if (decision.params.current) {
        expect(decision.params.current).toBeLessThanOrEqual(MAX_CURRENT_MA);
      }
    });
  });

  describe("computeStimFromCognitiveState", () => {
    it("should produce valid stim params", () => {
      const params = computeStimFromCognitiveState(0.3, 0.7);
      expect(params.current).toBeGreaterThanOrEqual(0.5);
      expect(params.current).toBeLessThanOrEqual(MAX_CURRENT_MA);
      expect(params.frequency).toBe(6.0); // fatigue > 0.5
      expect(params.mode).toBe("tACS");
    });

    it("should use tDCS when fatigue is low", () => {
      const params = computeStimFromCognitiveState(0.8, 0.2);
      expect(params.mode).toBe("tDCS");
      expect(params.waveform).toBe("dc");
    });

    it("should use tACS when fatigue is high", () => {
      const params = computeStimFromCognitiveState(0.3, 0.8);
      expect(params.mode).toBe("tACS");
      expect(params.waveform).toBe("ac");
    });
  });

  describe("detectArtifacts", () => {
    it("should return no artifacts on clean data", () => {
      const data = Array.from({ length: 8 }, () =>
        Array.from({ length: 1000 }, () => Math.sin(Math.random() * 0.1)),
      );
      const result = detectArtifacts(data, 250);
      expect(result.detected).toBe(false);
      expect(result.type).toBe("none");
    });

    it("should detect muscle artifacts (high amplitude + deviation)", () => {
      const data = Array.from({ length: 8 }, () =>
        Array.from({ length: 1000 }, () => Math.sin(Math.random() * 0.1)),
      );
      // Inject muscle artifact — amplitude > 100 with >10% deviation
      for (let i = 100; i < 200; i++) {
        data[0][i] = 150;
      }
      const result = detectArtifacts(data, 250);
      expect(result.detected).toBe(true);
    });

    it("should detect EOG artifacts (blink spike)", () => {
      const data = Array.from({ length: 8 }, () =>
        Array.from({ length: 1000 }, () => Math.sin(Math.random() * 0.1)),
      );
      // Inject EOG blink — amplitude > 75 with >5% deviation
      for (let i = 80; i < 180; i++) {
        data[0][i] = 80;
      }
      const result = detectArtifacts(data, 250);
      expect(result.detected).toBe(true);
    });

    it("should handle empty data gracefully", () => {
      const result = detectArtifacts([], 250);
      expect(result.detected).toBe(false);
    });

    it("should handle short channels gracefully", () => {
      const data = [[0.1, 0.2, 0.3]]; // Too short for window
      const result = detectArtifacts(data, 250);
      expect(result.detected).toBe(false);
    });

    it("should report affected channels", () => {
      const data = Array.from({ length: 8 }, () =>
        Array.from({ length: 1000 }, () => Math.sin(Math.random() * 0.1)),
      );
      // Inject artifact on channel 3 — amplitude > 100 with >10% deviation
      for (let i = 100; i < 200; i++) {
        data[3][i] = 150;
      }
      const result = detectArtifacts(data, 250);
      expect(result.channels).toContain(3);
    });
  });

  describe("Session Management", () => {
    it("should detect session timeout", () => {
      const session: StimSession = {
        sessionId: "test-1",
        device: { vendorId: "0x16c0", productId: "0x04b4", productName: "Test", maxCurrent: 2.0, supportedModes: ["tDCS"], isApproved: true },
        params: { mode: "tDCS", waveform: "dc", current: 1.0, frequency: 0, duration: 600, targetRegion: "dlPFC", montage: { anode: "F3", cathode: "F4" } },
        startTime: Date.now() - SESSION_TIMEOUT_MS - 1000,
        lastSafetyCheck: Date.now(),
        safetyState: "ok",
        events: [],
        impedanceChecks: 1,
        artifactDetections: 0,
      };
      expect(checkSessionTimeout(session)).toBe(true);
    });

    it("should not timeout before limit", () => {
      const session: StimSession = {
        sessionId: "test-1",
        device: { vendorId: "0x16c0", productId: "0x04b4", productName: "Test", maxCurrent: 2.0, supportedModes: ["tDCS"], isApproved: true },
        params: { mode: "tDCS", waveform: "dc", current: 1.0, frequency: 0, duration: 600, targetRegion: "dlPFC", montage: { anode: "F3", cathode: "F4" } },
        startTime: Date.now() - 1000,
        lastSafetyCheck: Date.now(),
        safetyState: "ok",
        events: [],
        impedanceChecks: 1,
        artifactDetections: 0,
      };
      expect(checkSessionTimeout(session)).toBe(false);
    });

    it("should log safety events", () => {
      const session: StimSession = {
        sessionId: "test-1",
        device: { vendorId: "0x16c0", productId: "0x04b4", productName: "Test", maxCurrent: 2.0, supportedModes: ["tDCS"], isApproved: true },
        params: { mode: "tDCS", waveform: "dc", current: 1.0, frequency: 0, duration: 600, targetRegion: "dlPFC", montage: { anode: "F3", cathode: "F4" } },
        startTime: Date.now(),
        lastSafetyCheck: Date.now(),
        safetyState: "ok",
        events: [],
        impedanceChecks: 1,
        artifactDetections: 0,
      };
      logSafetyEvent(session, "impedance_high", "Test event");
      expect(session.events).toHaveLength(1);
      expect(session.events[0].type).toBe("impedance_high");
    });
  });

  describe("Diagnostics", () => {
    it("should report correct max current", () => {
      const diag = getNeuroStimDiagnostics();
      expect(diag.maxCurrent).toBe(MAX_CURRENT_MA);
    });

    it("should report correct timeout", () => {
      const diag = getNeuroStimDiagnostics();
      expect(diag.sessionTimeoutMs).toBe(SESSION_TIMEOUT_MS);
    });

    it("should report correct impedance limit", () => {
      const diag = getNeuroStimDiagnostics();
      expect(diag.maxImpedance).toBe(MAX_IMPEDANCE_KOHM);
    });

    it("should report approved vendors", () => {
      const diag = getNeuroStimDiagnostics();
      expect(diag.approvedVendors.soterix).toBe(0x16c0);
      expect(diag.approvedVendors.openbci).toBe(0x2341);
    });
  });

  describe("Reset", () => {
    it("should reset without errors", () => {
      expect(() => resetNeuroStim()).not.toThrow();
    });
  });
});
