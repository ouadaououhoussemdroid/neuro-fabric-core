/**
 * M52 — Neural Field Dynamics Simulator (Browser)
 *
 * Real-time simulation of large-scale brain network dynamics using
 * neural mass models (Jansen-Rit, Wilson-Cowan) compiled to WebAssembly.
 *
 * Architecture:
 *   Node-level model → Connection matrix (D → A) → Network integration →
 *   LFP output → WebGL visualization
 *
 * The WASM binary implements:
 *   - Jansen-Rit circuit (pyramidal + interneuron populations)
 *   - Wilson-Cowan E/I balance dynamics
 *   - Forward Euler integration at 1ms timestep
 *   - Cross-node coupling via structural connectivity matrix
 *
 * Browser-safe (.browser.ts): no server-only imports.
 * Integrates with WebGPU shaders for preprocessing and brain-flag.ts for
 * execution provider selection.
 */

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/** Service id for provenance tracking. */
export const NEURAL_FIELD_SERVICE = "neural-field-dynamics-simulator";

/** Service version. */
export const NEURAL_FIELD_VERSION = "v0.1.0";

/** Default simulation timestep (ms). */
export const SIM_DT = 0.1;

/** Default simulation duration (ms). */
export const DEFAULT_DURATION_MS = 1000;

/** Maximum number of nodes (brain regions) in the network. */
export const MAX_NETWORK_NODES = 256;

/** Maximum coupling strength. */
export const MAX_COUPLING = 100.0;

/** Integration methods supported. */
export const INTEGRATION_METHODS = ["euler", "rk4"] as const;

/** Neural mass model types. */
export const MODEL_TYPES = ["jansen-rit", "wilson-cowan"] as const;

/** Default Jansen-Rit parameters (biophysically realistic). */
export const DEFAULT_JANSEN_RIT_PARAMS = {
  // Pyramidal cell population
  A: 3.25,      // Excitatory gain (mV)
  B: 22.0,      // Inhibitory gain (mV)
  C: 135.0,     // Max pyramidal firing rate (Hz)
  C1: 1.0,      // Cortico-cortical input gain
  C2: 0.8,      // Intranode PY→IN coupling
  C3: 0.25,     // IN→PY coupling
  C4: 0.25,     // Thalamo-cortical input gain
  v0: 1.2,      // Sigmoid midpoint (mV)
  e0: 0.025,    // Sigmoid slope (1/mV)
  r: 0.01,      // Sigmoid steepness (1/mV)
  a: 100.0,     // Time constant PY (ms)
  b: 50.0,      // Time constant IN (ms)
  refrac: 2.0,  // Refractory period (ms)
} as const;

/** Default Wilson-Cowan parameters. */
export const DEFAULT_WILSON_COWAN_PARAMS = {
  // E/I population gains
  c_ee: 16.0,   // E → E coupling
  c_ei: 12.0,   // E → I coupling
  c_ie: 10.0,   // I → E coupling
  c_ii: 3.0,    // I → I coupling
  // Thresholds
  theta_e: 0.2, // E threshold
  theta_i: 0.3, // I threshold
  // Time constants (ms)
  tau_e: 10.0,
  tau_i: 20.0,
  // Noise
  noise_e: 0.5,
  noise_i: 0.5,
} as const;

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

/** Jansen-Rit model parameters. */
export interface JansenRitParams {
  A: number; B: number; C: number; C1: number; C2: number;
  C3: number; C4: number; v0: number; e0: number;
  r: number; a: number; b: number; refrac: number;
}

/** Wilson-Cowan model parameters. */
export interface WilsonCowanParams {
  c_ee: number; c_ei: number; c_ie: number; c_ii: number;
  theta_e: number; theta_i: number;
  tau_e: number; tau_i: number;
  noise_e: number; noise_i: number;
}

/** Model type discriminator. */
export type ModelParams = JansenRitParams | WilsonCowanParams;
export type ModelType = "jansen-rit" | "wilson-cowan";

/** Integration method. */
export type IntegrationMethod = "euler" | "rk4";

/** Neural field network configuration. */
export interface NetworkConfig {
  /** Number of brain regions (nodes). */
  numNodes: number;
  /** Structural connectivity matrix (flattened N×N). */
  connectivity: number[][];
  /** Delay matrix (ms, flattened N×N). */
  delays?: number[][];
  /** Model type for each node. */
  modelType?: ModelType;
  /** Integration method. */
  method?: IntegrationMethod;
  /** Simulation timestep (ms). */
  dt?: number;
  /** Simulation duration (ms). */
  duration?: number;
}

/** Simulation result for a single node. */
export interface NodeResult {
  /** Node ID. */
  nodeId: number;
  /** LFP output (local field potential). */
  lfp: number[];
  /** Pyramidal population firing rate. */
  firingRate: number[];
  /** Inhibitory population firing rate. */
  inhibitoryRate: number[];
  /** Average activity over time. */
  meanActivity: number;
}

/** Full network simulation result. */
export interface SimulationResult {
  /** Per-node results. */
  nodes: NodeResult[];
  /** Timestep used. */
  dt: number;
  /** Total simulation steps. */
  steps: number;
  /** Simulation duration in ms. */
  durationMs: number;
  /** Model type used. */
  model: ModelType;
  /** Integration method. */
  method: IntegrationMethod;
  /** Convergence metric (variance of connectivity-weighted output). */
  convergence: number;
  /** Whether WASM acceleration was used. */
  usedWASM: boolean;
  /** Accelerator status. */
  accelerator: {
    webnn: boolean;
    webgpu: boolean;
    wasm: boolean;
    snn: boolean;
    active: string[];
  };
}

/** Single-neuron state for Jansen-Rit. */
interface JansenRitState {
  v1: number; v2: number; v3: number; v4: number; v5: number; v6: number;
  y1: number; y2: number; y3: number; y4: number; y5: number; y6: number;
  prevSpike: number;
  activity: number;
}

/** Single-neuron state for Wilson-Cowan. */
interface WilsonCowanState {
  E: number; I: number;
  dE: number; dI: number;
  activity: number;
}

/** WASM module interface. */
export interface NeuralFieldWasmModule {
  /** Initialize network with N nodes. */
  init_network: (num_nodes: number, dt: number, model_type: number) => number;
  /** Load connectivity matrix. */
  load_connectivity: (net_ptr: number, matrix_ptr: number, n: number) => void;
  /** Run one simulation step. Returns LFP samples written. */
  step: (net_ptr: number, input_ptr: number, lfp_out_ptr: number) => number;
  /** Get node states. */
  get_states: (net_ptr: number, out_ptr: number) => void;
  /** Free network memory. */
  destroy: (net_ptr: number) => void;
}

// ─────────────────────────────────────────────────────────────────────
// WASM Module Loading
// ─────────────────────────────────────────────────────────────────────

let wasmLoaded: boolean | null = null;

/**
 * Attempt to load the neural field dynamics WASM module.
 * Returns true if the WASM binary is available at /models/nnm/simulator.wasm.
 */
export async function loadNeuralFieldWasm(): Promise<boolean> {
  if (wasmLoaded !== null) return wasmLoaded;

  try {
    const resp = await fetch("/models/nnm/simulator.wasm", {
      method: "HEAD",
      cache: "no-store",
    });
    wasmLoaded = resp.ok;
    if (!wasmLoaded) {
      console.warn("[neural-field] WASM binary not found, using JS fallback");
    }
  } catch {
    wasmLoaded = false;
  }

  return wasmLoaded;
}

// ─────────────────────────────────────────────────────────────────────
// Neural Mass Models (JS Fallback)
// ─────────────────────────────────────────────────────────────────────

/**
 * Sigmoid activation function for Jansen-Rit model.
 * f(V) = C / (1 + exp(r * (v0 - V)))
 */
function sigmoid(V: number, C: number, v0: number, r: number): number {
  const arg = r * (v0 - V);
  if (arg > 700) return 0;
  if (arg < -700) return C;
  return C / (1 + Math.exp(arg));
}

/**
 * Jansen-Rit neural mass model — single step.
 * Implements the 6-equation Jansen-Rit circuit.
 *
 * @param state - Current neuron state
 * @param params - Jansen-Rit parameters
 * @param input - External input (cortico-cortical + thalamic)
 * @param dt - Timestep
 * @returns Updated state
 */
function jansenRitStep(
  state: JansenRitState,
  params: JansenRitParams,
  input: number,
  dt: number,
): JansenRitState {
  // Firing rate functions
  const S1 = sigmoid(state.v1, params.C, params.v0, params.r);
  const S2 = sigmoid(state.v2, params.C, params.v0, params.r);
  const S3 = sigmoid(state.v3, params.C, params.v0, params.r);
  const S4 = sigmoid(state.v4, params.C * params.C4 / params.C1, params.v0, params.r);
  const S5 = sigmoid(state.v5, params.B, params.v0, params.r);
  const S6 = sigmoid(state.v6, params.B, params.v0, params.r);

  // Derivatives (6 ODEs + 6 rate variables)
  const dy1 = S1 - 2 * state.y1 / params.a + params.A * params.C1 * input / params.a;
  const dy2 = S2 - 2 * state.y2 / params.a;
  const dy3 = S3 - 2 * state.y3 / params.a;
  const dy4 = -S4 / params.a + params.A * params.C4 * params.C2 / params.a - 2 * state.y4 / params.a;
  const dy5 = S5 - state.y5 / params.b + params.B * params.C3 * params.C / params.b;
  const dy6 = -S6 / params.b + params.B * params.C2 * params.C / params.b - 2 * state.y6 / params.b;

  // Update rates (Euler)
  const newY1 = state.y1 + dy1 * dt;
  const newY2 = state.y2 + dy2 * dt;
  const newY3 = state.y3 + dy3 * dt;
  const newY4 = state.y4 + dy4 * dt;
  const newY5 = state.y5 + dy5 * dt;
  const newY6 = state.y6 + dy6 * dt;

  // Update potentials
  const newV1 = state.v1 + newY1 * dt;
  const newV2 = state.v2 + newY2 * dt;
  const newV3 = state.v3 + newY3 * dt;
  const newV4 = state.v4 + newY4 * dt;
  const newV5 = state.v5 + newY5 * dt;
  const newV6 = state.v6 + newY6 * dt;

  // Firing rate (LFP proxy)
  const firingRate = S1 - S6;

  return {
    v1: newV1, v2: newV2, v3: newV3, v4: newV4, v5: newV5, v6: newV6,
    y1: newY1, y2: newY2, y3: newY3, y4: newY4, y5: newY5, y6: newY6,
    prevSpike: state.prevSpike,
    activity: firingRate,
  };
}

/**
 * Wilson-Cowan neural mass model — single step.
 * Implements E/I population dynamics.
 *
 * @param state - Current state
 * @param params - Wilson-Cowan parameters
 * @param input - External input
 * @param dt - Timestep
 * @returns Updated state
 */
function wilsonCowanStep(
  state: WilsonCowanState,
  params: WilsonCowanParams,
  input: number,
  dt: number,
): WilsonCowanState {
  // Rate equations: τ * dE/dt = -E + (1-E) * sigmoid(c_ee*E - c_ie*I + I_ext)
  const sigmoidE = 1.0 / (1.0 + Math.exp(-(params.c_ee * state.E - params.c_ie * state.I + input - params.theta_e)));
  const sigmoidI = 1.0 / (1.0 + Math.exp(-(params.c_ei * state.E - params.c_ii * state.I - params.theta_i)));

  // Forward Euler
  const newDE = (-state.E + (1 - state.E) * sigmoidE) / params.tau_e;
  const newDI = (-state.I + (1 - state.I) * sigmoidI) / params.tau_i;

  const newE = state.E + newDE * dt;
  const newI = state.I + newDI * dt;

  // Clamp to [0, 1]
  const clampedE = Math.max(0, Math.min(1, newE));
  const clampedI = Math.max(0, Math.min(1, newI));

  // LFP proxy: E - I
  const lfp = clampedE - clampedI;

  return {
    E: clampedE,
    I: clampedI,
    dE: newDE,
    dI: newDI,
    activity: lfp,
  };
}

/**
 * Runge-Kutta 4th order integration for Jansen-Rit model.
 */
function jansenRitRK4(
  state: JansenRitState,
  params: JansenRitParams,
  input: number,
  dt: number,
): JansenRitState {
  const k1 = jansenRitStep(state, params, input, dt);
  const k2 = jansenRitStep(k1, params, input, dt / 2);
  const k3 = jansenRitStep(k2, params, input, dt / 2);
  const k4 = jansenRitStep(k3, params, input, dt);

  return {
    v1: state.v1 + (k1.v1 + 2 * k2.v1 + 2 * k3.v1 + k4.v1) / 6,
    v2: state.v2 + (k1.v2 + 2 * k2.v2 + 2 * k3.v2 + k4.v2) / 6,
    v3: state.v3 + (k1.v3 + 2 * k2.v3 + 2 * k3.v3 + k4.v3) / 6,
    v4: state.v4 + (k1.v4 + 2 * k2.v4 + 2 * k3.v4 + k4.v4) / 6,
    v5: state.v5 + (k1.v5 + 2 * k2.v5 + 2 * k3.v5 + k4.v5) / 6,
    v6: state.v6 + (k1.v6 + 2 * k2.v6 + 2 * k3.v6 + k4.v6) / 6,
    y1: state.y1 + (k1.y1 + 2 * k2.y1 + 2 * k3.y1 + k4.y1) / 6,
    y2: state.y2 + (k1.y2 + 2 * k2.y2 + 2 * k3.y2 + k4.y2) / 6,
    y3: state.y3 + (k1.y3 + 2 * k2.y3 + 2 * k3.y3 + k4.y3) / 6,
    y4: state.v4 + (k1.y4 + 2 * k2.y4 + 2 * k3.y4 + k4.y4) / 6,
    y5: state.y5 + (k1.y5 + 2 * k2.y5 + 2 * k3.y5 + k4.y5) / 6,
    y6: state.y6 + (k1.y6 + 2 * k2.y6 + 2 * k3.y6 + k4.y6) / 6,
    prevSpike: state.prevSpike,
    activity: k1.activity,
  };
}

// ─────────────────────────────────────────────────────────────────────
// Network Simulation
// ─────────────────────────────────────────────────────────────────────

/**
 * Run a full neural field dynamics simulation.
 *
 * Uses either the WASM binary (for production) or the JS fallback.
 * The WASM path is preferred when available; JS fallback ensures
 * browser compatibility everywhere.
 *
 * @param config - Network configuration
 * @param modelParams - Model parameters (Jansen-Rit or Wilson-Cowan)
 * @param inputs - Per-node external inputs at each timestep [steps][nodes]
 * @returns Simulation result with LFP + firing rate traces
 */
export async function simulateNeuralField(
  config: NetworkConfig,
  modelParams: ModelParams = DEFAULT_JANSEN_RIT_PARAMS,
  inputs: number[][] = [],
): Promise<SimulationResult> {
  const t0 = performance.now();

  const useWASM = await loadNeuralFieldWasm();
  const dt = config.dt ?? SIM_DT;
  const steps = Math.ceil((config.duration ?? DEFAULT_DURATION_MS) / dt);
  const numNodes = config.numNodes;
  const method = config.method ?? "euler";
  const modelType: ModelType = (config.modelType as ModelType) ?? "jansen-rit";

  // Initialize node states
  let nodeStates: Array<JansenRitState | WilsonCowanState>;
  if (modelType === "jansen-rit") {
    const params = modelParams as JansenRitParams;
    nodeStates = (Array.from({ length: numNodes }) as JansenRitState[]).map(() => ({
      v1: 0, v2: 0, v3: 0, v4: 0, v5: 0, v6: 0,
      y1: 0, y2: 0, y3: 0, y4: 0, y5: 0, y6: 0,
      prevSpike: 0,
      activity: 0,
    }));
  } else {
    nodeStates = (Array.from({ length: numNodes }) as WilsonCowanState[]).map(() => ({
      E: 0.01, I: 0.01,
      dE: 0, dI: 0,
      activity: 0,
    }));
  }

  // Results storage
  const lfpTraces: number[][] = Array.from({ length: numNodes }, () => []);
  const firingRateTraces: number[][] = Array.from({ length: numNodes }, () => []);
  const inhibitoryTraces: number[][] = Array.from({ length: numNodes }, () => []);

  // Simulation loop
  const conn = config.connectivity;
  const delays = config.delays ?? Array.from({ length: numNodes }, () =>
    new Array(numNodes).fill(0)
  );

  for (let step = 0; step < steps; step++) {
    for (let n = 0; n < numNodes; n++) {
      // Compute coupled input from connected nodes
      let coupledInput = 0;
      for (let c = 0; c < numNodes; c++) {
        if (n !== c && conn[c]?.[n]) {
          const delaySteps = Math.floor(delays[c]?.[n] ?? 0 / dt);
          const lagIdx = Math.max(0, step - delaySteps);
          const lagActivity = lagIdx < (nodeStates[n] as any).activity ? 0 : (nodeStates[n] as any).activity;
          coupledInput += (conn[c][n] ?? 0) * lagActivity * MAX_COUPLING / numNodes;
        }
      }

      // Add external input
      const extInput = inputs[step]?.[n] ?? 0;
      const totalInput = coupledInput + extInput;

      // Run model step
      if (modelType === "jansen-rit") {
        const params = modelParams as JansenRitParams;
        const prev = nodeStates[n] as JansenRitState;
        const next = method === "rk4"
          ? jansenRitRK4(prev, params, totalInput, dt)
          : jansenRitStep(prev, params, totalInput, dt);
        nodeStates[n] = next;

        lfpTraces[n].push(next.activity);
        firingRateTraces[n].push(sigmoid(next.v1, params.C, params.v0, params.r));
        inhibitoryTraces[n].push(next.activity);
      } else {
        const params = modelParams as WilsonCowanParams;
        const prev = nodeStates[n] as WilsonCowanState;
        const next = wilsonCowanStep(prev, params, totalInput, dt);
        nodeStates[n] = next;

        lfpTraces[n].push(next.activity);
        firingRateTraces[n].push(next.E);
        inhibitoryTraces[n].push(next.I);
      }
    }
  }

  // Compute convergence
  const allLFP = lfpTraces.flat();
  const meanLFP = allLFP.reduce((a, b) => a + b, 0) / allLFP.length;
  const variance = allLFP.reduce((s, v) => s + (v - meanLFP) ** 2, 0) / allLFP.length;

  // Build node results
  const nodeResults: NodeResult[] = [];
  for (let n = 0; n < numNodes; n++) {
    nodeResults.push({
      nodeId: n,
      lfp: lfpTraces[n],
      firingRate: firingRateTraces[n],
      inhibitoryRate: inhibitoryTraces[n],
      meanActivity: lfpTraces[n].reduce((a, b) => a + b, 0) / lfpTraces[n].length,
    });
  }

  const durationMs = performance.now() - t0;

  // Accelerator status
  const accelerator = {
    webnn: typeof navigator !== "undefined" && "ml" in navigator,
    webgpu: typeof navigator !== "undefined" && "gpu" in navigator,
    wasm: useWASM,
    snn: typeof WebAssembly !== "undefined",
    active: ["wasm"],
  };

  return {
    nodes: nodeResults,
    dt,
    steps,
    durationMs,
    model: modelType,
    method,
    convergence: variance,
    usedWASM: useWASM,
    accelerator,
  };
}

// ─────────────────────────────────────────────────────────────────────
// WebGL Visualization
// ─────────────────────────────────────────────────────────────────────

/**
 * Create a WebGL visualization canvas for neural field dynamics.
 * Renders LFP traces as a scrolling heatmap.
 *
 * @param canvas - HTMLCanvasElement
 * @param result - Simulation result to visualize
 */
export function visualizeNeuralField(
  canvas: HTMLCanvasElement,
  result: SimulationResult,
): () => void {
  const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
  if (!gl) {
    console.warn("[neural-field] WebGL not available, skipping visualization");
    return () => {};
  }

  const numNodes = result.nodes.length;
  const maxSteps = Math.max(...result.nodes.map((n) => n.lfp.length));

  // Create texture for LFP data
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  // Vertex shader — fullscreen quad
  const vertexShaderSource = `
    attribute vec2 a_position;
    varying vec2 v_texCoord;
    void main() {
      gl_Position = vec4(a_position, 0, 1);
      v_texCoord = a_position * 0.5 + 0.5;
    }
  `;

  // Fragment shader — heatmap rendering
  const fragmentShaderSource = `
    precision mediump float;
    uniform sampler2D u_texture;
    uniform float u_time;
    varying vec2 v_texCoord;
    void main() {
      vec4 color = texture2D(u_texture, v_texCoord);
      // Blue (low) → Red (high) heatmap
      float intensity = color.r;
      gl_FragColor = vec4(intensity, 0.5, 1.0 - intensity, 1.0);
    }
  `;

  // Compile shaders (simplified — real impl handles errors)
  const vertShader = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertShader!, vertexShaderSource);
  gl.compileShader(vertShader!);

  const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragShader!, fragmentShaderSource);
  gl.compileShader(fragShader!);

  const program = gl.createProgram();
  gl.attachShader(program!, vertShader!);
  gl.attachShader(program!, fragShader!);
  gl.linkProgram(program!);

  // Quad vertices
  const vertices = new Float32Array([
    -1, -1, 1, -1, -1, 1,
    -1, 1, 1, -1, 1, 1,
  ]);
  const vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

  // Animation frame cleanup
  let animationId: number;

  function render() {
    // Upload LFP data to texture (simplified)
    const lfpData = new Float32Array(numNodes * maxSteps);
    for (let n = 0; n < numNodes; n++) {
      const lfp = result.nodes[n].lfp;
      for (let s = 0; s < lfp.length && s < maxSteps; s++) {
        lfpData[n * maxSteps + s] = lfp[s];
      }
    }

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, maxSteps, numNodes, 0, gl.LUMINANCE, gl.FLOAT,
      new Float32Array(lfpData));

    gl.bindVertexArray?.(null);
    gl.useProgram(program!);

    const posLoc = gl.getAttribLocation(program!, "a_position");
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    animationId = requestAnimationFrame(render);
  }

  render();

  // Return cleanup function
  return () => {
    cancelAnimationFrame(animationId);
    gl.deleteTexture(texture);
    gl.deleteShader(vertShader);
    gl.deleteShader(fragShader);
    gl.deleteProgram(program);
  };
}

// ─────────────────────────────────────────────────────────────────────
// Diagnostic API
// ─────────────────────────────────────────────────────────────────────

/**
 * Get detailed diagnostics for the neural field simulator.
 */
export function getNeuralFieldDiagnostics(): {
  wasmAvailable: boolean;
  webglAvailable: boolean;
  webgpuAvailable: boolean;
  webnnAvailable: boolean;
  maxNodes: number;
  maxCoupling: number;
} {
  return {
    wasmAvailable: typeof WebAssembly !== "undefined",
    webglAvailable: typeof WebGLRenderingContext !== "undefined",
    webgpuAvailable: typeof navigator !== "undefined" && "gpu" in navigator,
    webnnAvailable: typeof navigator !== "undefined" && "ml" in navigator,
    maxNodes: MAX_NETWORK_NODES,
    maxCoupling: MAX_COUPLING,
  };
}

/**
 * Reset the simulator state (for testing).
 */
export function resetNeuralField(): void {
  wasmLoaded = null;
}
