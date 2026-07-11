/**
 * Production Braindecode bridge backed by ONNX Runtime Web.
 *
 * Rationale (see docs/adr/0001-braindecode-execution-strategy.md):
 * Braindecode is a PyTorch library and cannot execute directly in a browser
 * tab. Pyodide + a PyTorch wheel is technically possible but ships ~80 MB of
 * runtime, lacks WebGPU/threading parity, and breaks SSR. Exporting the
 * trained Braindecode model to ONNX once (offline) and running it through the
 * already-wired `ONNXAdapter` gives us:
 *
 *   - real neural inference in the browser (WASM SIMD, optional WebGPU),
 *   - identical artefact format to every other ONNX model in the registry,
 *   - automatic capability detection + fallback to PCA via the existing facade,
 *   - zero extra runtime download on the SSR path.
 *
 * This module exposes `createONNXBraindecodeBridge()`, which satisfies the
 * `BraindecodeBridge` contract by delegating to an internal `ONNXAdapter`.
 * The Braindecode adapter then validates window shape and forwards through it.
 */
import { ONNXAdapter, isONNXRuntimeAvailable, type ONNXAdapterOptions } from "./onnx-adapter";
import type {
  BraindecodeBridge,
  BraindecodeForwardResult,
  BraindecodeModelName,
} from "./braindecode-adapter";

export interface ONNXBraindecodeBridgeOptions {
  /** URL, ArrayBuffer, or bytes of the exported ONNX file. */
  artifact: ONNXAdapterOptions["artifact"];
  architecture: BraindecodeModelName;
  channels: number;
  sampleRate: number;
  windowSamples: number;
  embeddingDim: number;
  /** Optional output tensor name override (logits head). */
  logitsOutputName?: string;
  /** Optional output tensor name for the embedding head. */
  embeddingOutputName?: string;
  executionProviders?: ONNXAdapterOptions["executionProviders"];
  /** Injected runtime (for tests). */
  runtime?: ONNXAdapterOptions["runtime"];
}

export function createONNXBraindecodeBridge(opts: ONNXBraindecodeBridgeOptions): BraindecodeBridge {
  let embeddingAdapter: ONNXAdapter | null = null;
  let logitsAdapter: ONNXAdapter | null = null;

  const baseAdapter = (outputName: string | undefined, id: string) =>
    new ONNXAdapter({
      id,
      name: `Braindecode/${opts.architecture} (ONNX)`,
      version: "0.1.0",
      description: `Braindecode ${opts.architecture} exported to ONNX`,
      artifact: opts.artifact,
      task: "embedding",
      inputShape: {
        kind: "raw",
        channels: opts.channels,
        samples: opts.windowSamples,
      },
      outputName,
      channels: opts.channels,
      sampleRate: opts.sampleRate,
      windowSamples: opts.windowSamples,
      embeddingDim: opts.embeddingDim,
      executionProviders: opts.executionProviders,
      runtime: opts.runtime,
    });

  return {
    async isAvailable() {
      return isONNXRuntimeAvailable(opts.runtime);
    },
    async load() {
      if (!embeddingAdapter) {
        embeddingAdapter = baseAdapter(opts.embeddingOutputName, "bd-onnx-emb");
        await embeddingAdapter.load();
      }
      if (opts.logitsOutputName && !logitsAdapter) {
        logitsAdapter = baseAdapter(opts.logitsOutputName, "bd-onnx-logits");
        await logitsAdapter.load();
      }
    },
    async forward(window): Promise<BraindecodeForwardResult> {
      if (!embeddingAdapter) {
        throw new Error("ONNX Braindecode bridge: not loaded");
      }
      const input = {
        kind: "windows" as const,
        windows: [
          {
            data: window,
            sampleRate: opts.sampleRate,
            start: 0,
            end: opts.windowSamples,
          },
        ],
      };
      const emb = await embeddingAdapter.embed!(input);
      const result: BraindecodeForwardResult = { embedding: emb.vector };
      if (logitsAdapter) {
        const logits = await logitsAdapter.embed!(input);
        result.logits = logits.vector;
      }
      return result;
    },
    async unload() {
      await embeddingAdapter?.unload();
      await logitsAdapter?.unload();
      embeddingAdapter = null;
      logitsAdapter = null;
    },
  };
}
