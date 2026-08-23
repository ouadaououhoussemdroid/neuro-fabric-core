/**
 * M-Scientific-Reboot — Consistency Gate Tests
 *
 * These tests enforce the scientific claims freeze and prevent stale/invalid
 * metrics from being presented as scientifically validated.
 *
 * Gate rules:
 * 1. Every task head MUST have a scientificStatus field
 * 2. PROXY_DEMONSTRATION heads MUST have previousMetrics.status === "INVALID"
 * 3. BLOCKED heads MUST have previousMetrics.status === "INVALID"
 * 4. SCIENTIFICALLY_VALIDATED heads have genuine ground-truth labels
 * 5. experimentId MUST point to actual training runs, not seed runs
 */
import { describe, it, expect, beforeAll } from "vitest";
import {
  COGNITIVE_HEADS,
  COGNITIVE_LINEAR_PROBE_JOINT_2312,
  COGNITIVE_LINEAR_PROBE_V2_32,
  registerCognitiveHeads,
} from "../cognitive.registry";
import {
  ANOMALY_HEADS,
  ANOMALY_MAHALANOBIS_PROBE_JOINT_2312,
  ANOMALY_MAHALANOBIS_PROBE_V2_32,
  registerAnomalyHeads,
} from "../anomaly.registry";
import {
  SLEEP_HEADS,
  SLEEP_STAGING_PROBE_JOINT_2312,
  SLEEP_QUALITY_PROBE_JOINT_2312,
  SLEEP_STAGING_PROBE_V2_32,
  SLEEP_QUALITY_PROBE_V2_32,
  registerSleepHeads,
} from "../sleep.registry";

describe("Scientific Status — Consistency Gate", () => {
  beforeAll(() => {
    registerCognitiveHeads();
    registerAnomalyHeads();
    registerSleepHeads();
  });

  describe("Cognitive heads", () => {
    it("Joint-2312 probe has scientificStatus set", () => {
      expect(COGNITIVE_LINEAR_PROBE_JOINT_2312.scientificStatus).toBeDefined();
    });

    it("Joint-2312 probe is SCIENTIFICALLY_VALIDATED with genuine labels", () => {
      expect(COGNITIVE_LINEAR_PROBE_JOINT_2312.scientificStatus).toBe("SCIENTIFICALLY_VALIDATED");
      expect(COGNITIVE_LINEAR_PROBE_JOINT_2312.training.dataset).toContain("EEGMMIDB");
      expect(COGNITIVE_LINEAR_PROBE_JOINT_2312.previousMetrics?.status).toBe("INVALID");
      expect(COGNITIVE_LINEAR_PROBE_JOINT_2312.experimentId).toBe("m33-scientific-reboot");
    });

    it("Joint-2312 probe does NOT use proxy labels", () => {
      expect(COGNITIVE_LINEAR_PROBE_JOINT_2312.training.protocol).not.toContain("proxy");
      expect(COGNITIVE_LINEAR_PROBE_JOINT_2312.training.dataset).not.toMatch(/proxy/i);
    });

    it("V2-32 fallback is reclassified as PROXY_DEMONSTRATION", () => {
      expect(COGNITIVE_LINEAR_PROBE_V2_32.scientificStatus).toBe("PROXY_DEMONSTRATION");
      expect(COGNITIVE_LINEAR_PROBE_V2_32.previousMetrics?.status).toBe("INVALID");
    });

    it("experimentId points to scientific-reboot, not seed run", () => {
      expect(COGNITIVE_LINEAR_PROBE_JOINT_2312.experimentId).not.toBe("m33-cognitive-workload-probe");
      expect(COGNITIVE_LINEAR_PROBE_JOINT_2312.experimentId).toBe("m33-scientific-reboot");
    });
  });

  describe("Anomaly heads", () => {
    it("Joint-2312 probe has scientificStatus set", () => {
      expect(ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.scientificStatus).toBeDefined();
    });

    it("Joint-2312 probe methodology is consistent (Mahalanobis CV = ONNX)", () => {
      expect(ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.scientificStatus).toBe("EXPERIMENTAL");
      expect(ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.previousMetrics?.status).toBe("INVALID");
      expect(ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.experimentId).toBe("m34-anomaly-detection-probe-v2");
    });

    it("V2-32 fallback is reclassified as PROXY_DEMONSTRATION", () => {
      expect(ANOMALY_MAHALANOBIS_PROBE_V2_32.scientificStatus).toBe("PROXY_DEMONSTRATION");
      expect(ANOMALY_MAHALANOBIS_PROBE_V2_32.previousMetrics?.status).toBe("INVALID");
    });
  });

  describe("Sleep heads", () => {
    it("Staging probe is BLOCKED (no Sleep-EDF data)", () => {
      expect(SLEEP_STAGING_PROBE_JOINT_2312.scientificStatus).toBe("BLOCKED");
      expect(SLEEP_STAGING_PROBE_JOINT_2312.previousMetrics?.status).toBe("INVALID");
      expect(SLEEP_STAGING_PROBE_JOINT_2312.experimentId).toBe("m38-sleep-staging-blocked");
    });

    it("Quality probe is BLOCKED (no Sleep-EDF data)", () => {
      expect(SLEEP_QUALITY_PROBE_JOINT_2312.scientificStatus).toBe("BLOCKED");
      expect(SLEEP_QUALITY_PROBE_JOINT_2312.previousMetrics?.status).toBe("INVALID");
      expect(SLEEP_QUALITY_PROBE_JOINT_2312.experimentId).toBe("m38-sleep-quality-blocked");
    });

    it("V2-32 fallbacks are reclassified as PROXY_DEMONSTRATION", () => {
      expect(SLEEP_STAGING_PROBE_V2_32.scientificStatus).toBe("PROXY_DEMONSTRATION");
      expect(SLEEP_QUALITY_PROBE_V2_32.scientificStatus).toBe("PROXY_DEMONSTRATION");
    });
  });

  describe("No stale experimentIds", () => {
    it("cognitive: no experimentId points to seed run m39/m40", () => {
      for (const head of COGNITIVE_HEADS) {
        expect(head.experimentId).not.toBe("m39-sleep-staging-probe");
        expect(head.experimentId).not.toBe("m40-sleep-quality-probe");
      }
    });

    it("sleep: all experimentIds use m38 prefix (blocked status)", () => {
      for (const head of SLEEP_HEADS) {
        expect(head.experimentId).toMatch(/^m38-/);
      }
    });
  });

  describe("All heads have scientific classification", () => {
    it("every cognitive head has a scientificStatus", () => {
      for (const head of COGNITIVE_HEADS) {
        expect(head.scientificStatus).toBeDefined();
        expect(head.scientificStatus).toMatch(
          /SCIENTIFICALLY_VALIDATED|ENGINEERING_VALIDATED|EXPERIMENTAL|PROXY_DEMONSTRATION|BLOCKED/,
        );
      }
    });

    it("every anomaly head has a scientificStatus", () => {
      for (const head of ANOMALY_HEADS) {
        expect(head.scientificStatus).toBeDefined();
      }
    });

    it("every sleep head has a scientificStatus", () => {
      for (const head of SLEEP_HEADS) {
        expect(head.scientificStatus).toBeDefined();
      }
    });
  });
});
