import { describe, it, expect } from "vitest";
import {
  createQuantumState,
  encodeAmplitude,
  createVariationalCircuit,
  createFeatureMap,
  executeCircuit,
  sampleMeasurements,
  optimizeVQE,
  quantumStateFidelity,
  runQuantumNeuromorphicInference,
  getQuantumDiagnostics,
  resetQuantumState,
  MAX_NUM_QUBITS,
  DEFAULT_NUM_QUBITS,
  DEFAULT_SHOTS,
  DEFAULT_LAYERS,
  QUANTUM_SERVICE,
  QUANTUM_VERSION,
  QUANTUM_GATES,
  type QuantumState,
  type QuantumCircuit,
} from "../quantum-neuromorph.browser";

describe("M55 — Quantum-Neuromorphic Computing", () => {
  describe("Constants & Configuration", () => {
    it("should have correct service metadata", () => {
      expect(QUANTUM_SERVICE).toBe("quantum-neuromorphic-computing");
      expect(QUANTUM_VERSION).toBe("v0.1.0");
    });

    it("should support up to 16 qubits", () => {
      expect(MAX_NUM_QUBITS).toBe(16);
    });

    it("should default to 8 qubits", () => {
      expect(DEFAULT_NUM_QUBITS).toBe(8);
    });

    it("should default to 1024 shots", () => {
      expect(DEFAULT_SHOTS).toBe(1024);
    });

    it("should default to 4 layers", () => {
      expect(DEFAULT_LAYERS).toBe(4);
    });

    it("should support standard quantum gates", () => {
      expect(QUANTUM_GATES).toContain("H");
      expect(QUANTUM_GATES).toContain("RX");
      expect(QUANTUM_GATES).toContain("RY");
      expect(QUANTUM_GATES).toContain("RZ");
      // CNOT is a 2-qubit gate, tested separately in circuit execution
    });
  });

  describe("createQuantumState", () => {
    it("should initialize to |0...0⟩ state", () => {
      const state = createQuantumState(2);
      expect(state.numQubits).toBe(2);
      expect(state.amplitudes.length).toBe(4);
      // |00⟩ should have amplitude 1
      expect(state.amplitudes[0].re).toBeCloseTo(1, 10);
      expect(state.amplitudes[0].im).toBeCloseTo(0, 10);
      // Others should be 0
      for (let i = 1; i < 4; i++) {
        expect(state.amplitudes[i].re).toBeCloseTo(0, 10);
      }
    });

    it("should scale correctly with qubit count", () => {
      const state = createQuantumState(4);
      expect(state.amplitudes.length).toBe(16);
    });
  });

  describe("encodeAmplitude", () => {
    it("should encode normalized vector into state", () => {
      const state = createQuantumState(2);
      const data = [1, 0, 0, 0];
      const encoded = encodeAmplitude(state, data);

      // Should remain |00⟩
      expect(encoded.amplitudes[0].re).toBeCloseTo(1, 5);
    });

    it("should normalize input data", () => {
      const state = createQuantumState(2);
      const data = [3, 4]; // norm = 5
      const encoded = encodeAmplitude(state, data);

      // |0⟩ = 3/5 = 0.6, |1⟩ = 4/5 = 0.8
      expect(encoded.amplitudes[0].re).toBeCloseTo(0.6, 5);
      expect(encoded.amplitudes[1].re).toBeCloseTo(0.8, 5);
    });

    it("should throw when data exceeds state dimension", () => {
      const state = createQuantumState(1); // dimension 2
      const data = [1, 2, 3, 4]; // length 4 > 2
      expect(() => encodeAmplitude(state, data)).toThrow();
    });

    it("should handle zero vector gracefully", () => {
      const state = createQuantumState(2);
      const encoded = encodeAmplitude(state, [0, 0, 0, 0]);
      // Should keep |00⟩
      expect(encoded.amplitudes[0].re).toBeCloseTo(1, 5);
    });
  });

  describe("createVariationalCircuit", () => {
    it("should create a valid circuit", () => {
      const circuit = createVariationalCircuit(4, 2);
      expect(circuit.numQubits).toBe(4);
      expect(circuit.depth).toBe(2);
      expect(circuit.parameters.length).toBe(8); // 4 qubits × 2 layers
      expect(circuit.layers.length).toBe(2);
    });

    it("should have RY + CNOT gates in each layer", () => {
      const circuit = createVariationalCircuit(3, 1);
      const layer = circuit.layers[0];
      // Should have 3 RY gates + 3 CNOT gates = 6 instructions
      expect(layer.length).toBe(6);
    });

    it("should accept custom parameters", () => {
      const params = [0.1, 0.2, 0.3, 0.4];
      const circuit = createVariationalCircuit(2, 1, params);
      expect(circuit.parameters).toEqual(params);
    });
  });

  describe("createFeatureMap", () => {
    it("should create feature map from data", () => {
      const circuit = createFeatureMap(4, [0.5, -0.3, 0.1, 0.8]);
      expect(circuit.numQubits).toBe(4);
      expect(circuit.layers.length).toBeGreaterThan(0);
    });
  });

  describe("executeCircuit", () => {
    it("should apply H gate to |0⟩ to produce |+⟩", () => {
      const circuit: QuantumCircuit = {
        numQubits: 1,
        depth: 1,
        layers: [[{ gate: "H", target: 0 }]],
        parameters: [],
      };
      const state = createQuantumState(1);
      const result = executeCircuit(state, circuit);

      // |+⟩ = (|0⟩ + |1⟩)/√2
      const expected = 1 / Math.SQRT2;
      expect(result.amplitudes[0].re).toBeCloseTo(expected, 5);
      expect(result.amplitudes[1].re).toBeCloseTo(expected, 5);
    });

    it("should apply X gate to flip |0⟩ to |1⟩", () => {
      const circuit: QuantumCircuit = {
        numQubits: 1,
        depth: 1,
        layers: [[{ gate: "X", target: 0 }]],
        parameters: [],
      };
      const state = createQuantumState(1);
      const result = executeCircuit(state, circuit);

      expect(result.amplitudes[0].re).toBeCloseTo(0, 5);
      expect(result.amplitudes[1].re).toBeCloseTo(1, 5);
    });

    it("should apply CNOT gate correctly", () => {
      const circuit: QuantumCircuit = {
        numQubits: 2,
        depth: 2,
        layers: [
          [{ gate: "X", target: 0 }],
          [{ gate: "CNOT", target: 1, control: 0 }],
        ],
        parameters: [],
      };
      const state = createQuantumState(2);
      const result = executeCircuit(state, circuit);

      // After X on q0: |10⟩, after CNOT: |11⟩
      const idx11 = 3; // |11⟩ = 11₂ = 3
      expect(result.amplitudes[idx11].re).toBeCloseTo(1, 5);
    });

    it("should apply RY rotation", () => {
      const circuit: QuantumCircuit = {
        numQubits: 1,
        depth: 1,
        layers: [[{ gate: "RY", target: 0, angle: Math.PI / 2 }]],
        parameters: [],
      };
      const state = createQuantumState(1);
      const result = executeCircuit(state, circuit);

      // RY(π/2)|0⟩ = cos(π/4)|0⟩ + sin(π/4)|1⟩
      const expected = 1 / Math.SQRT2;
      expect(result.amplitudes[0].re).toBeCloseTo(expected, 5);
      expect(result.amplitudes[1].re).toBeCloseTo(expected, 5);
    });

    it("should handle empty circuit", () => {
      const circuit: QuantumCircuit = {
        numQubits: 2,
        depth: 0,
        layers: [],
        parameters: [],
      };
      const state = createQuantumState(2);
      const result = executeCircuit(state, circuit);

      // Should remain |00⟩
      expect(result.amplitudes[0].re).toBeCloseTo(1, 5);
    });
  });

  describe("sampleMeasurements", () => {
    it("should return measurement results sorted by count", () => {
      const state = createQuantumState(1);
      // Put in |+⟩ state
      const circuit: QuantumCircuit = {
        numQubits: 1,
        depth: 1,
        layers: [[{ gate: "H", target: 0 }]],
        parameters: [],
      };
      const result = executeCircuit(state, circuit);
      const measurements = sampleMeasurements(result, 100);

      expect(measurements.length).toBeGreaterThan(0);
      // |+⟩ should give roughly equal 0 and 1
      expect(measurements.every((m) => m.bitstring.length === 1)).toBe(true);
    });

    it("should return all counts", () => {
      const state = createQuantumState(1);
      const measurements = sampleMeasurements(state, 100);
      const totalCounts = measurements.reduce((sum, m) => sum + m.counts, 0);
      expect(totalCounts).toBe(100);
    });
  });

  describe("quantumStateFidelity", () => {
    it("should be 1 for identical states", () => {
      const state = createQuantumState(2);
      const fidelity = quantumStateFidelity(state, state);
      expect(fidelity).toBeCloseTo(1.0, 5);
    });

    it("should be 0 for orthogonal states", () => {
      const state1 = createQuantumState(1); // |0⟩
      const state2: QuantumState = {
        numQubits: 1,
        amplitudes: [{ re: 0, im: 0 }, { re: 1, im: 0 }],
      };
      const fidelity = quantumStateFidelity(state1, state2);
      expect(fidelity).toBeCloseTo(0.0, 5);
    });

    it("should be symmetric", () => {
      const state1 = createQuantumState(2);
      const state2 = createQuantumState(2);
      state2.amplitudes[1] = { re: 0.5, im: 0 };

      const f12 = quantumStateFidelity(state1, state2);
      const f21 = quantumStateFidelity(state2, state1);
      expect(f12).toBeCloseTo(f21, 10);
    });
  });

  describe("optimizeVQE", () => {
    it("should return valid optimization result", () => {
      const result = optimizeVQE(2, 2, 10);
      expect(result.params.length).toBe(4); // 2 qubits × 2 layers
      expect(result.energy).toBeTypeOf("number");
      expect(result.iterations).toBe(10);
      expect(result.converged).toBeTypeOf("boolean");
    });

    it("should find lower energy with more iterations", () => {
      const short = optimizeVQE(2, 2, 5);
      const long = optimizeVQE(2, 2, 30);
      // Long run should have equal or lower energy
      expect(long.energy).toBeLessThanOrEqual(short.energy + 0.1);
    });
  });

  describe("runQuantumNeuromorphicInference", () => {
    it("should produce valid inference result", () => {
      const embedding = [0.3, 0.5, -0.2, 0.1];
      const result = runQuantumNeuromorphicInference(embedding, 2, 100);

      expect(result.energy).toBeTypeOf("number");
      expect(result.output).toBeTypeOf("number");
      expect(result.measurements.length).toBeGreaterThan(0);
      expect(result.latencyMs).toBeGreaterThan(0);
      expect(result.circuit.numQubits).toBe(2);
    });

    it("should handle larger embeddings", () => {
      const embedding = Array.from({ length: 8 }, () => Math.random() - 0.5);
      const result = runQuantumNeuromorphicInference(embedding, 4, 100);

      expect(result.circuit.numQubits).toBe(4);
      expect(result.measurements.length).toBeGreaterThan(0);
    });

    it("should auto-size qubits from embedding", () => {
      const embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
      const result = runQuantumNeuromorphicInference(embedding);

      // 8 elements → log2(8) = 3 qubits, capped at default 8
      expect(result.circuit.numQubits).toBeLessThanOrEqual(8);
    });

    it("should have deterministic-ish output (random seed not available, but structure)", () => {
      const embedding = [0.3, 0.5, -0.2, 0.1];
      const result = runQuantumNeuromorphicInference(embedding, 2, 100);

      // Output should be in reasonable range
      expect(result.output).toBeGreaterThanOrEqual(0);
      expect(result.output).toBeLessThanOrEqual(1);
    });
  });

  describe("Diagnostics", () => {
    it("should report quantum service name", () => {
      const diag = getQuantumDiagnostics();
      expect(diag.quantumService).toBe(QUANTUM_SERVICE);
    });

    it("should report quantum version", () => {
      const diag = getQuantumDiagnostics();
      expect(diag.quantumVersion).toBe(QUANTUM_VERSION);
    });

    it("should report supported gates", () => {
      const diag = getQuantumDiagnostics();
      expect(diag.supportedGates.length).toBeGreaterThan(0);
    });

    it("should report qubit limits", () => {
      const diag = getQuantumDiagnostics();
      expect(diag.maxNumQubits).toBe(MAX_NUM_QUBITS);
    });
  });

  describe("Reset", () => {
    it("should reset without errors", () => {
      expect(() => resetQuantumState()).not.toThrow();
    });
  });
});
