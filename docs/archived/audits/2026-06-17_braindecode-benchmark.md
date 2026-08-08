# Braindecode EEGConformer — Benchmark Report

- Date: 2026-06-17
- Harness: `src/lib/ai/benchmark/index.ts` (`benchmarkAll`)
- Input: synthetic 22-channel, 1000-sample, 250 Hz window
  (multi-sine, deterministic seed)
- Iterations: 20 per adapter
- Environment: reference workstation, Chrome 124, WASM SIMD enabled
  (no WebGPU); single-tab, idle.

## Results

| Adapter                          | Dim | Mean (ms) | P50 (ms) | P95 (ms) | Heap Δ (MB) | Notes                                  |
| -------------------------------- | --: | --------: | -------: | -------: | ----------: | -------------------------------------- |
| `pca-legacy-v1`                  |  64 |       3.1 |      2.9 |      4.4 |         0.1 | Closed-form, JS only                   |
| Generic ONNX (placeholder model) |  32 |      18.6 |     17.9 |     25.2 |         5.4 | Same shape as EEGConformer             |
| `braindecode-eegconformer-prod`  |  32 |     312.8 |    304.5 |    402.1 |        18.7 | First-call load ~520 ms (cached after) |

## Interpretation

- EEGConformer is ~100× slower than PCA at single-window granularity,
  but produces a representation transferable across paradigms — see the
  selection report for the quality trade-off.
- After the first inference the model session is cached for the page
  lifetime; subsequent calls are dominated by tensor copies, not load.
- Peak heap stays under 25 MB, comfortably below the 250 MB SPA budget.

## Methodology notes

- All adapters share the same `embed()` facade, so latency includes
  validation + L2 normalisation.
- `heap delta` uses `performance.memory.usedJSHeapSize` where exposed
  (Chromium-only); blank where unavailable.
- "Generic ONNX" row uses a synthetic 22×1000→32 model identical in
  shape to EEGConformer so the gap reflects model complexity only.

## Reproducing

```ts
import { benchmarkAll } from "@/lib/ai/benchmark";
import { registerBraindecodeEEGConformer } from "@/lib/ai/models/registry";

registerBraindecodeEEGConformer({ artifact: { kind: "url", url: "/models/eegconformer.onnx" } });
const rows = await benchmarkAll(["pca-legacy-v1", "braindecode-eegconformer-prod"], input, 20);
console.table(rows);
```
