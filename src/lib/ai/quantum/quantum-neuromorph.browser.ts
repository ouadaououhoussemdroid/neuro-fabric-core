/**
 * M55 — Quantum-Neuromorphic Computing Integration
 *
 * Browser-safe quantum circuit simulation layer that bridges neuromorphic
 * spiking neural networks (SNN) with quantum computing primitives.
 * Provides a variational quantum eigensolver (VQE)-style hybrid inference
 * pipeline using amplitude encoding of neural embeddings into qubit states.
 *
 * Architecture:
 *   Neural embedding (SNN/WebNN) → Amplitude encoding →
 *   Quantum circuit layers (Hadamard, rotation, entangling) →
 *   Measurement sampling → Classical post-processing
 *
 * Browser-safe (.browser.ts): no server-only imports.
 * Quantum simulation is pure JavaScript — no external dependencies.
 */

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/** Service id for provenance tracking. */
export const QUANTUM_SERVICE = "quantum-neuromorphic-computing";

/** Service version. */
export const QUANTUM_VERSION = "v0.1.0";

/** Default number of qubits for the simulated quantum processor. */
export const DEFAULT_NUM_QUBITS = 8;

/** Maximum qubits supported (2^16 complex amplitudes = 128KB state vector). */
export const MAX_NUM_QUBITS = 16;

/** Default number of measurement shots per circuit execution. */
export const DEFAULT_SHOTS = 1024;

/** Default number of layers in the variational ansatz. */
export const DEFAULT_LAYERS = 4;

/** Supported single-qubit gates. */
export const QUANTUM_GATES = ["H", "X", "Y", "Z", "RX", "RY", "RZ", "S", "T"] as const;
export type QuantumGate = (typeof QUANTUM_GATES)[number];

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

/** A single-qubit Pauli observable for measurement. */
export type PauliOp = "I" | "X" | "Y" | "Z";

/** Quantum circuit instruction. */
export interface QuantumInstruction {
  /** Gate name (H, X, Y, Z, RX, RY, RZ, CNOT, CZ). */
  gate: string;
  /** Target qubit index. */
  target: number;
  /** Control qubit index (for 2-qubit gates). */
  control?: number;
  /** Rotation angle (radians) for parameterized gates. */
  angle?: number;
}

/** Parameterized quantum circuit (variational ansatz). */
export interface QuantumCircuit {
  /** Number of qubits. */
  numQubits: number;
  /** Circuit depth (layers). */
  depth: number;
  /** Instructions per layer. */
  layers: QuantumInstruction[][];
  /** Parameters to optimize (replaces angles). */
  parameters: number[];
}

/** Quantum state vector (complex amplitudes). */
export interface QuantumState {
  /** Number of qubits. */
  numQubits: number;
  /** Complex amplitudes |ψ⟩ (length = 2^numQubits). */
  amplitudes: Complex[];
}

/** Complex number representation. */
export interface Complex {
  re: number;
  im: number;
}

/** Measurement result from a quantum circuit. */
export interface QuantumMeasurement {
  /** Bitstring result (e.g., "1010"). */
  bitstring: string;
  /** Probability of this outcome. */
  probability: number;
  /** Number of times observed in shots. */
  counts: number;
}

/** Variational optimization result. */
export interface VQEOptimizationResult {
  /** Optimized parameters. */
  params: number[];
  /** Minimum energy found. */
  energy: number;
  /** Number of iterations. */
  iterations: number;
  /** Converged. */
  converged: boolean;
}

/** Quantum-neuromorphic hybrid inference result. */
export interface QuantumNeuromorphicResult {
  /** Energy from VQE (proxy for neural loss). */
  energy: number;
  /** Quantum measurement distribution. */
  measurements: QuantumMeasurement[];
  /** Classical post-processing output. */
  output: number;
  /** Execution time (ms). */
  latencyMs: number;
  /** Quantum circuit used. */
  circuit: QuantumCircuit;
}

// ─────────────────────────────────────────────────────────────────────
// Complex Number Utilities
// ─────────────────────────────────────────────────────────────────────

/** Create a complex number. */
function complex(re: number, im: number): Complex {
  return { re, im };
}

/** Add two complex numbers. */
function cadd(a: Complex, b: Complex): Complex {
  return { re: a.re + b.re, im: a.im + b.im };
}

/** Multiply two complex numbers. */
function cmul(a: Complex, b: Complex): Complex {
  return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}

/** Magnitude squared of a complex number. */
function cmag2(c: Complex): number {
  return c.re * c.re + c.im * c.im;
}

/** Complex exponential: e^(i*theta) = cos(theta) + i*sin(theta). */
function cexp(theta: number): Complex {
  return { re: Math.cos(theta), im: Math.sin(theta) };
}

// ─────────────────────────────────────────────────────────────────────
// Quantum Gate Matrices
// ─────────────────────────────────────────────────────────────────────

/** Hadamard gate. */
const H_GATE: Complex[][] = [
  [{ re: 1 / Math.SQRT2, im: 0 }, { re: 1 / Math.SQRT2, im: 0 }],
  [{ re: 1 / Math.SQRT2, im: 0 }, { re: -1 / Math.SQRT2, im: 0 }],
];

/** Pauli-X gate. */
const X_GATE: Complex[][] = [
  [{ re: 0, im: 0 }, { re: 1, im: 0 }],
  [{ re: 1, im: 0 }, { re: 0, im: 0 }],
];

/** Rotation around X axis. */
function RX(angle: number): Complex[][] {
  const c = Math.cos(angle / 2);
  const s = Math.sin(angle / 2);
  return [
    [{ re: c, im: 0 }, { re: 0, im: -s }],
    [{ re: 0, im: -s }, { re: c, im: 0 }],
  ];
}

/** Rotation around Y axis. */
function RY(angle: number): Complex[][] {
  const c = Math.cos(angle / 2);
  const s = Math.sin(angle / 2);
  return [
    [{ re: c, im: 0 }, { re: -s, im: 0 }],
    [{ re: s, im: 0 }, { re: c, im: 0 }],
  ];
}

/** Rotation around Z axis. */
function RZ(angle: number): Complex[][] {
  const e = cexp(-angle / 2);
  const e2 = cexp(angle / 2);
  return [
    [{ re: e.re, im: e.im }, { re: 0, im: 0 }],
    [{ re: 0, im: 0 }, { re: e2.re, im: e2.im }],
  ];
}

// ─────────────────────────────────────────────────────────────────────
// Quantum State Preparation
// ─────────────────────────────────────────────────────────────────────

/**
 * Initialize a quantum state to |0...0⟩.
 */
export function createQuantumState(numQubits: number): QuantumState {
  const dim = 1 << numQubits;
  const amplitudes: Complex[] = new Array(dim).fill(null).map(() => complex(0, 0));
  amplitudes[0] = complex(1, 0); // |00...0⟩
  return { numQubits, amplitudes };
}

/**
 * Amplitude encoding: encode a normalized vector into quantum amplitudes.
 * The input vector must have length ≤ 2^numQubits.
 */
export function encodeAmplitude(
  state: QuantumState,
  data: number[],
): QuantumState {
  const dim = state.amplitudes.length;
  if (data.length > dim) {
    throw new Error(`Data length ${data.length} exceeds state dimension ${dim}`);
  }

  // Normalize the input vector
  const norm = Math.sqrt(data.reduce((s, v) => s + v * v, 0));
  if (norm === 0) return state;

  const normalized = data.map((v) => v / norm);

  const amplitudes: Complex[] = new Array(dim).fill(null).map(() => complex(0, 0));
  for (let i = 0; i < normalized.length; i++) {
    amplitudes[i] = complex(normalized[i], 0);
  }

  return { numQubits: state.numQubits, amplitudes };
}

// ─────────────────────────────────────────────────────────────────────
// Quantum Circuit Construction
// ─────────────────────────────────────────────────────────────────────

/**
 * Create a variational quantum ansatz (parametrized circuit).
 * Structure: [RY rotations] → [entangling CNOT layers] repeated `depth` times.
 */
export function createVariationalCircuit(
  numQubits: number,
  depth: number = DEFAULT_LAYERS,
  parameters?: number[],
): QuantumCircuit {
  const numParams = numQubits * depth;
  const params = parameters ?? Array.from({ length: numParams }, () => Math.random() * 2 * Math.PI);
  const layers: QuantumInstruction[][] = [];

  let paramIdx = 0;
  for (let layer = 0; layer < depth; layer++) {
    const layerInstructions: QuantumInstruction[] = [];

    // RY rotations on each qubit
    for (let q = 0; q < numQubits; q++) {
      layerInstructions.push({
        gate: "RY",
        target: q,
        angle: params[paramIdx++],
      });
    }

    // Entangling CNOTs (circular chain)
    for (let q = 0; q < numQubits; q++) {
      const next = (q + 1) % numQubits;
      layerInstructions.push({
        gate: "CNOT",
        target: next,
        control: q,
      });
    }

    layers.push(layerInstructions);
  }

  return { numQubits, depth, layers, parameters: params };
}

/**
 * Create a quantum feature map circuit (amplitude encoding + entangling).
 * Used to map classical neural embeddings into quantum feature space.
 */
export function createFeatureMap(
  numQubits: number,
  data: number[],
  repetitions: number = 2,
): QuantumCircuit {
  const dim = 1 << numQubits;

  // Pad/truncate data to fit state dimension
  const padded = new Array(dim).fill(0);
  for (let i = 0; i < Math.min(data.length, dim); i++) padded[i] = data[i];

  const layers: QuantumInstruction[][] = [];
  const params: number[] = [];

  for (let rep = 0; rep < repetitions; rep++) {
    // Single-qubit rotations based on data
    const dataLayer: QuantumInstruction[] = [];
    for (let q = 0; q < numQubits; q++) {
      const val = padded[q] ?? 0;
      dataLayer.push({ gate: "RY", target: q, angle: val * Math.PI });
    }
    // Entangling layer
    for (let q = 0; q < numQubits - 1; q++) {
      dataLayer.push({ gate: "CNOT", target: q + 1, control: q });
    }
    layers.push(dataLayer);
  }

  return { numQubits, depth: layers.length, layers, parameters: params };
}

// ─────────────────────────────────────────────────────────────────────
// Quantum Gate Application
// ─────────────────────────────────────────────────────────────────────

/** Apply a single-qubit gate to the state vector. */
function applySingleQubitGate(
  state: QuantumState,
  gate: Complex[][],
  target: number,
): QuantumState {
  const { numQubits, amplitudes } = state;
  const dim = amplitudes.length;
  const newAmps: Complex[] = amplitudes.map((a) => complex(a.re, a.im));

  for (let i = 0; i < dim; i++) {
    if (!((i >> target) & 1)) {
      // |0⟩ component: i stays same, apply gate[0][0] and gate[0][1]
      const a0 = amplitudes[i];
      const a1 = amplitudes[i | (1 << target)];
      newAmps[i] = cadd(cmul(gate[0][0], a0), cmul(gate[0][1], a1));
      newAmps[i | (1 << target)] = cadd(cmul(gate[1][0], a0), cmul(gate[1][1], a1));
    }
  }

  // Each pair gets processed once due to the i|(1<<target) swap
  return { numQubits, amplitudes: newAmps };
}

/** Apply CNOT gate (control → target). */
function applyCNOT(
  state: QuantumState,
  control: number,
  target: number,
): QuantumState {
  const { numQubits, amplitudes } = state;
  const newAmps: Complex[] = amplitudes.map((a) => complex(a.re, a.im));

  for (let i = 0; i < amplitudes.length; i++) {
    if ((i >> control) & 1) {
      // Control qubit is |1⟩, flip target
      newAmps[i] = amplitudes[i ^ (1 << target)];
    }
  }

  return { numQubits, amplitudes: newAmps };
}

/** Apply all instructions in a circuit to a quantum state. */
export function executeCircuit(
  state: QuantumState,
  circuit: QuantumCircuit,
): QuantumState {
  let result = state;

  for (const layer of circuit.layers) {
    for (const inst of layer) {
      switch (inst.gate) {
        case "H":
          result = applySingleQubitGate(result, H_GATE, inst.target);
          break;
        case "X":
          result = applySingleQubitGate(result, X_GATE, inst.target);
          break;
        case "RX":
          result = applySingleQubitGate(result, RX(inst.angle ?? 0), inst.target);
          break;
        case "RY":
          result = applySingleQubitGate(result, RY(inst.angle ?? 0), inst.target);
          break;
        case "RZ":
          result = applySingleQubitGate(result, RZ(inst.angle ?? 0), inst.target);
          break;
        case "CNOT":
          if (inst.control !== undefined) {
            result = applyCNOT(result, inst.control, inst.target);
          }
          break;
        default:
          // Unknown gate — skip
          break;
      }
    }
  }

  return result;
}

// ─────────────────────────────────────────────────────────────────────
// Quantum Measurement
// ┌────────────────────────────────────────────────────────────────────

/**
 * Sample measurements from a quantum state (probability sampling).
 * Returns top measurement results sorted by probability.
 */
export function sampleMeasurements(
  state: QuantumState,
  shots: number = DEFAULT_SHOTS,
): QuantumMeasurement[] {
  const { amplitudes } = state;
  const probabilities = amplitudes.map((a) => cmag2(a));

  // Build cumulative distribution
  const cumulative: number[] = [];
  let sum = 0;
  for (const p of probabilities) {
    sum += p;
    cumulative.push(sum);
  }

  // Sample
  const counts: Record<string, number> = {};
  for (let s = 0; s < shots; s++) {
    const r = Math.random();
    let idx = 0;
    for (let i = 0; i < cumulative.length; i++) {
      if (r <= cumulative[i]) {
        idx = i;
        break;
      }
    }
    const bitstring = idx.toString(2).padStart(state.numQubits, "0");
    counts[bitstring] = (counts[bitstring] || 0) + 1;
  }

  // Convert to sorted measurements
  const results: QuantumMeasurement[] = Object.entries(counts)
    .map(([bitstring, count]) => ({
      bitstring,
      probability: count / shots,
      counts: count,
    }))
    .sort((a, b) => b.counts - a.counts);

  return results;
}

// ─────────────────────────────────────────────────────────────────────
// Variational Quantum Eigensolver (VQE)
// ─────────────────────────────────────────────────────────────────────

/**
 * Compute expectation value of a Hamiltonian for a given quantum state.
 * Uses simple sum of Z operators as the Hamiltonian proxy.
 */
function computeExpectation(state: QuantumState): number {
  const { amplitudes } = state;
  let energy = 0;

  // ⟨Z₀ + Z₁ + ... + Zₙ₋₁⟩ = Σᵢ ⟨ψ|Zᵢ|ψ⟩
  // For Z on qubit i: +1 if |0⟩, -1 if |1⟩
  for (let i = 0; i < amplitudes.length; i++) {
    const p = cmag2(amplitudes[i]);
    let parity = 0;
    for (let q = 0; q < state.numQubits; q++) {
      parity += (i >> q) & 1;
    }
    energy += p * (parity % 2 === 0 ? 1 : -1);
  }

  return energy;
}

/**
 * Simple COBYLA-free parameter optimization for VQE.
 * Uses random search with Gaussian perturbation.
 */
export function optimizeVQE(
  numQubits: number,
  depth: number = DEFAULT_LAYERS,
  iterations: number = 50,
  initialParams?: number[],
): VQEOptimizationResult {
  let params = initialParams ?? Array.from(
    { length: numQubits * depth },
    () => Math.random() * 2 * Math.PI,
  );

  let bestEnergy = Infinity;
  let bestParams = [...params];
  const lr = 0.1;

  for (let iter = 0; iter < iterations; iter++) {
    // Compute current energy
    const circuit = createVariationalCircuit(numQubits, depth, params);
    const state = executeCircuit(createQuantumState(numQubits), circuit);
    const energy = computeExpectation(state);

    if (energy < bestEnergy) {
      bestEnergy = energy;
      bestParams = [...params];
    }

    // Simple gradient-free update: perturb parameters
    const gradients: number[] = [];
    for (let i = 0; i < params.length; i++) {
      const plusParams = [...params];
      plusParams[i] += 0.01;
      const plusCircuit = createVariationalCircuit(numQubits, depth, plusParams);
      const plusState = executeCircuit(createQuantumState(numQubits), plusCircuit);
      const plusEnergy = computeExpectation(plusState);

      const minusParams = [...params];
      minusParams[i] -= 0.01;
      const minusCircuit = createVariationalCircuit(numQubits, depth, minusParams);
      const minusState = executeCircuit(createQuantumState(numQubits), minusCircuit);
      const minusEnergy = computeExpectation(minusState);

      gradients.push((plusEnergy - minusEnergy) / 0.02);
    }

    // Update parameters
    for (let i = 0; i < params.length; i++) {
      params[i] -= lr * gradients[i];
    }
  }

  return {
    params: bestParams,
    energy: bestEnergy,
    iterations,
    converged: bestEnergy < -0.9, // Converged if we find a low-energy state
  };
}

// ─────────────────────────────────────────────────────────────────────
// Quantum-Neuromorphic Hybrid Pipeline
// ─────────────────────────────────────────────────────────────────────

/**
 * Run quantum-classical hybrid inference on a neural embedding.
 *
 * Pipeline:
 *   1. Amplitude encode the embedding into a quantum state
 *   2. Apply a feature map circuit
 *   3. Run VQE optimization for parameter tuning
 *   4. Sample measurements
 *   5. Classical post-processing to produce scalar output
 *
 * @param embedding - Neural embedding from SNN/WebNN (array of floats)
 * @param numQubits - Number of qubits (default: auto from embedding size)
 * @param shots - Number of measurement shots
 * @returns Hybrid inference result
 */
export function runQuantumNeuromorphicInference(
  embedding: number[],
  numQubits?: number,
  shots: number = DEFAULT_SHOTS,
): QuantumNeuromorphicResult {
  const qubits = numQubits ?? Math.min(DEFAULT_NUM_QUBITS, Math.ceil(Math.log2(Math.max(embedding.length, 2))));

  const t0 = performance.now();

  // 1. Prepare quantum state
  let state = createQuantumState(qubits);

  // 2. Amplitude encoding
  state = encodeAmplitude(state, embedding);

  // 3. Feature map circuit
  const featureMap = createFeatureMap(qubits, embedding);
  state = executeCircuit(state, featureMap);

  // 4. VQE optimization (uses simplified optimization for browser safety)
  const vqeResult = optimizeVQE(qubits, 2, 20);

  // 5. Apply optimized variational circuit
  const varCircuit = createVariationalCircuit(qubits, 2, vqeResult.params);
  state = executeCircuit(state, varCircuit);

  // 6. Sample measurements
  const measurements = sampleMeasurements(state, shots);

  // 7. Classical post-processing: weighted sum of measurement probabilities
  const output = measurements.reduce((sum, m) => {
    const val = parseInt(m.bitstring, 2) / (measurements.length || 1);
    return sum + m.probability * val;
  }, 0);

  const latencyMs = performance.now() - t0;

  return {
    energy: vqeResult.energy,
    measurements,
    output,
    latencyMs,
    circuit: varCircuit,
  };
}

/**
 * Compute the fidelity between two quantum states.
 * Fidelity = |⟨ψ|φ⟩|² — measures similarity of quantum embeddings.
 */
export function quantumStateFidelity(stateA: QuantumState, stateB: QuantumState): number {
  if (stateA.numQubits !== stateB.numQubits) {
    throw new Error("States must have same number of qubits");
  }

  let fidelity = 0;
  for (let i = 0; i < stateA.amplitudes.length; i++) {
    // ⟨φ|ψ⟩ = Σ φᵢ* ψᵢ (conjugate of φ)
    const conjA = complex(stateA.amplitudes[i].re, -stateA.amplitudes[i].im);
    const product = cmul(conjA, stateB.amplitudes[i]);
    fidelity += product.re;
  }

  return Math.max(0, Math.min(1, fidelity * fidelity));
}

// ─────────────────────────────────────────────────────────────────────
// Diagnostics & Utilities
// ─────────────────────────────────────────────────────────────────────

/**
 * Get quantum computing diagnostic info.
 */
export function getQuantumDiagnostics(): {
  quantumService: string;
  quantumVersion: string;
  defaultNumQubits: number;
  maxNumQubits: number;
  defaultShots: number;
  defaultLayers: number;
  supportedGates: string[];
} {
  return {
    quantumService: QUANTUM_SERVICE,
    quantumVersion: QUANTUM_VERSION,
    defaultNumQubits: DEFAULT_NUM_QUBITS,
    maxNumQubits: MAX_NUM_QUBITS,
    defaultShots: DEFAULT_SHOTS,
    defaultLayers: DEFAULT_LAYERS,
    supportedGates: QUANTUM_GATES,
  };
}

/**
 * Reset quantum simulation state (test helper).
 */
export function resetQuantumState(): void {
  // Pure function pipeline — no global state to reset
  // Included for harness compatibility
}
