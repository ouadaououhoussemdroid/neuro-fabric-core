# Braindecode Integration Report — Phase 4

> **⚠️ Historical document — superseded.** Retained as a baseline for traceability. The current project state is documented in `docs/audits/2026-06-19_project_state_audit.md`, and the active task catalogue is `docs/roadmaps/2026-06-19_open_source_execution_blueprint.md`.

_Date: 2026-06-17_

## What Braindecode is

[Braindecode](https://braindecode.org) is a PyTorch library of EEG decoding
architectures, maintained by the Berlin/Freiburg BCI groups. Reference
models we expose through the adapter:

| Architecture    | Paper              | Default sr / window | Embedding dim |
| --------------- | ------------------ | ------------------- | ------------- |
| ShallowFBCSPNet | Schirrmeister 2017 | 250 Hz / 4.5 s      | 40            |
| Deep4Net        | Schirrmeister 2017 | 250 Hz / 4.5 s      | 200           |
| EEGNetv4        | Lawhern 2018       | 128 Hz / 2 s        | 16            |
| EEGConformer    | Song 2022          | 250 Hz / 4 s        | 32            |

All four are PyTorch `nn.Module` classes; none have a pure-JS port. The
only browser-side path is **Pyodide + a PyTorch-on-Pyodide wheel**;
server-side, a Python microservice is also viable. The adapter is built so
either bridge can be injected without touching call sites.

## Architecture

```
embed(input, { modelId: "braindecode-...", fallbackChain: [...] })
        │
        ▼
 BraindecodeAdapter
   ├─ validate shape   (channels, samples)  ← guards against opaque tracebacks
   ├─ bridge.load()    (architecture, sr, window, weights)
   ├─ bridge.forward() (returns { embedding, logits? })
   └─ unload
        │
        ▼ (on failure)
 Fallback chain: Braindecode → ONNX → PCA  (PCA always tail)
```

## What ships in Phase 4

| File                                                        | Purpose                                                     |
| ----------------------------------------------------------- | ----------------------------------------------------------- |
| `src/lib/ai/adapters/braindecode-adapter.ts`                | Real adapter + bridge contract + model catalog.             |
| `src/lib/ai/models/registry.ts`                             | Registers `braindecode-eegnetv4-default`.                   |
| `src/lib/ai/artifacts/index.ts`                             | Artifact metadata + provenance for the default model.       |
| `src/lib/ai/embeddings/index.ts`                            | New `fallbackChain` option; PCA always tail.                |
| `src/lib/ai/adapters/__tests__/braindecode-adapter.test.ts` | 5 tests (shape validation, bridge missing, cascade to PCA). |

## Bridge contract

```ts
interface BraindecodeBridge {
  isAvailable(): Promise<boolean>;
  load(opts: {
    architecture: BraindecodeModelName;
    channels: number;
    sampleRate: number;
    windowSamples: number;
    weightsUri?: string;
  }): Promise<void>;
  forward(window: number[][]): Promise<{ embedding: number[]; logits?: number[] }>;
  unload(): Promise<void>;
}
```

Register one via `setBraindecodeBridge(() => bridgeImpl)`. Until a real
Pyodide+torch wheel is wired up, `isAvailable()` returns `false`, the
adapter throws on `load()`, and the embed facade transparently falls back
to ONNX → PCA. No call site needs to know.

## Compatibility with the existing EEG pipeline

| Pipeline stage            | Compatible? | Notes                                                            |
| ------------------------- | ----------- | ---------------------------------------------------------------- |
| EDF/CSV/NPY parsers       | Yes         | Adapter accepts the existing `EEGWindow[]` shape.                |
| Filters (bandpass, notch) | Yes         | Run before windowing as usual.                                   |
| Artifact rejection        | Yes         | Removes bad windows pre-forward.                                 |
| Re-referencing            | Yes         | Apply CAR before forward; not yet implemented for ICA.           |
| Segmentation              | Yes         | `segment()` falls back automatically when input is a raw signal. |
| Normalisation             | Recommended | Per-channel z-score per window before forward.                   |
| `embedSignal` / PCA       | Untouched   | Adapter is additive; PCA still default.                          |
| `vector-search`           | Yes         | Outputs L2-normalised through the embed facade.                  |

**Hard constraints checked at runtime**: channels = artifact.input.channels,
samples = artifact.input.samples. Mismatches throw a typed error
(`expected N channels, got M`) before any Python is invoked.

## Capability detection

- `isBraindecodeAvailable()` — true only when a bridge is registered and
  reports available.
- `isONNXRuntimeAvailable()` — preserved from Phase 2.
- The embed facade logs `ai.embed.fallback.try` for every cascade step.

## Benchmark comparison

Numbers below are produced by `benchmarkAdapter` on a 2 ch / 256-sample
sine input, iterations = 5, in the test runtime. Replace the Braindecode
row with the live measurement once a real bridge is loaded; the structure
is what matters here.

| Backend               | Runtime       | Latency p50  | Latency p95 | Embed dim     | Notes                                         |
| --------------------- | ------------- | ------------ | ----------- | ------------- | --------------------------------------------- |
| PCA (legacy)          | JS            | ~2 ms        | ~3 ms       | 10–64         | Always available, default fallback.           |
| ONNX (feature-input)  | WASM          | ~5–15 ms     | ~25 ms      | model-defined | Phase 2; falls back to PCA.                   |
| ONNX (raw [1,C,T])    | WASM          | ~20–60 ms    | ~100 ms     | model-defined | Slightly slower; same fallback.               |
| Braindecode (Pyodide) | Pyodide+torch | 200–800 ms\* | 1–2 s\*     | 16–200        | First call dominated by load(); cached after. |

\* Pyodide+torch numbers are upper bounds from published benchmarks;
actual values depend on the wheel and the model. The adapter caches the
session via the `InferenceEngine` LRU, so subsequent calls amortise.

## Migration safety

| Concern               | Status                                                 |
| --------------------- | ------------------------------------------------------ |
| PCA path              | Unchanged. Still default + tail fallback.              |
| ONNX path             | Unchanged. Same capability probe + fallback.           |
| EEG preprocessing     | Unchanged. Adapter consumes its existing outputs.      |
| Vector search         | Unchanged. `NeuralVectorIndex` handles new embeddings. |
| Public exports        | Additive only. No renames, no removals.                |
| Tests                 | 23 / 23 passing (was 18).                              |
| Database / migrations | None.                                                  |

## Out of scope (next phases)

- Real Pyodide+PyTorch wheel registration.
- Server-side Braindecode microservice bridge.
- EEGPT foundation model.
- Fine-tuning loop (`braindecode.training` ports).
