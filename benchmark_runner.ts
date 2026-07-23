/**
 * Stand-alone benchmark runner for the production validation task.
 * It measures latency and embedding properties for the two
 * competing models: pca-legacy-v1 and braindecode-eegconformer-prod.
 *
 * The script:
 *   1. Ensures the model registry is initialised.
 *   2. Builds a dummy EEG input (22 channels, 1000 samples @ 250 Hz).
 *   3. Calls benchmarkAll for each model (5 iterations).
 *   4. Logs the raw result object as JSON.
 *
 * Run with: npx ts-node benchmark_runner.ts
 */

import { benchmarkAll } from './src/lib/ai/benchmark';
import './src/lib/ai/models/registry'; // side-effect: populates the model registry

/** Build a ModelInput with the required shape. */
function makeInput(channels: number, samples: number, sr: number) {
  const data: number[][] = [];
  for (let c = 0; c < channels; c++) {
    const ch = new Array<number>(samples);
    for (let t = 0; t < samples; t++) {
      ch[t] = Math.sin((2 * Math.PI * (c + 1) * t) / sr);
    }
    data.push(ch);
  }
  return { kind: 'windows', windows: [{ data, sampleRate: sr, start: 0, end: samples }] };
}

// Execute
(async () => {
  const input = makeInput(22, 1000, 250);
  // Run 5 iterations per model (as used by the library)
  const results = await benchmarkAll(['pca-legacy-v1', 'braindecode-eegconformer-prod'], input, 5);
  console.log(JSON.stringify(results, null, 2));
})().catch(err => {
  console.error('Benchmark failed:', err);
  process.exit(1);
}
