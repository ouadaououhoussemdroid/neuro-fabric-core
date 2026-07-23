import { benchmarkAll } from './src/lib/ai/benchmark';
import './src/lib/ai/models/registry.js';

function makeInput(channels, samples, sr) {
  const data = [];
  for (let c = 0; c < channels; c++) {
    const ch = new Array(samples);
    for (let t = 0; t < samples; t++) {
      ch[t] = Math.sin((2 * Math.PI * (c + 1) * t) / sr);
    }
    data.push(ch);
  }
  return { kind: 'windows', windows: [{ data, sampleRate: sr, start: 0, end: samples }] };
}

(async () => {
  const input = makeInput(22, 1000, 250);
  const results = await benchmarkAll(['pca-legacy-v1', 'braindecode-eegconformer-prod'], input, 5);
  console.log(JSON.stringify(results, null, 2));
})().catch(err => {
  console.error('Benchmark failed:', err);
  process.exit(1);
}
