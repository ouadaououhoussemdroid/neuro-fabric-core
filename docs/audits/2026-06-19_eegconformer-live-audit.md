# EEGConformer Live Registration Audit

- Date: 2026-06-19
- Baseline: docs/audits/2026-06-17_braindecode-production-readiness.md
- Current: working tree on this date
- Scope: read-only. Verify that `eegconformer.onnx` is correctly wired
  into the `embed()` routing path and confirm the updated maturity score.

## 1. Registration check

`src/lib/ai/models/registry.ts` contains, at module top-level (after the
`PCAEmbeddingAdapter`, `PyTorchExportAdapter`, `EEGPTAdapter`, and default
`BraindecodeAdapter` registrations):

```ts
// Production EEGConformer — ONNX artefact served from /models/
registerBraindecodeEEGConformer({ artifact: "/models/eegconformer.onnx" });
/** Default embedder used when callers do not pin a model id. */
export const DEFAULT_EMBEDDER_ID = "pca-legacy-v1";
```

Confirmed: the `registerBraindecodeEEGConformer()` call appears
immediately before the `DEFAULT_EMBEDDER_ID` export, and the artefact
URL is `/models/eegconformer.onnx`. Registration runs at module import
time, so by the time any caller reaches the registry the
`braindecode-eegconformer-prod` id is present.

## 2. Routing check

Trace:

1. Caller invokes `embedEEG(input, opts?)` from
   `src/lib/ai/inference/embed-eeg.ts`.
2. `embedEEG` resolves `preferred = opts.preferredModelId ??
"braindecode-eegnetv4-onnx"`. When the caller does not override and
   `hasModel(preferred)` is false, `startId` falls through to the first
   entry in `chain` (PCA). To route through EEGConformer, callers
   should pass `preferredModelId: "braindecode-eegconformer-prod"`;
   the current registry registers
   `braindecode-eegconformer-prod`, **not** `braindecode-eegnetv4-onnx`.
3. `embedEEG` calls `embed(input, { modelId: startId, fallbackChain,
fallbackToPCA: true, ... })` from `src/lib/ai/embeddings/index.ts`.
4. `embed()` calls `createAdapter(id)` from
   `src/lib/ai/models/registry.ts`, which returns a
   `BraindecodeAdapter` whose bridge is the ONNX bridge constructed by
   `createONNXBraindecodeBridge({ artifact: "/models/eegconformer.onnx",
architecture: "EEGConformer", channels: 22, sampleRate: 250,
windowSamples: 1000, embeddingDim: 32, embeddingOutputName:
"embedding" })`.
5. On adapter `load()` / `embed()` failure (no WASM runtime, fetch
   error, parity issue), `embed()` walks `fallbackChain` and finally
   `pca-legacy-v1`, returning `{ fellBack: true, fallbackReason }`.

Full fallback chain as wired today:

```
braindecode-eegconformer-prod   (only when caller passes preferredModelId)
  → <opts.onnxModelId>          (only if provided and registered)
  → pca-legacy-v1               (always available)
```

Caveat — `embedEEG()`'s default `preferredModelId` is still
`"braindecode-eegnetv4-onnx"` (DEFAULT_PREFERRED in
`src/lib/ai/inference/embed-eeg.ts`). That id is **not** registered in
the current registry, so default callers will silently start on PCA.
Routing through the EEGConformer artefact requires either updating
`DEFAULT_PREFERRED` or every caller passing the prod id explicitly.
This is the single open wiring gap.

## 3. Artifact check

`public/models/eegconformer.onnx` exists in the repo.

- Size: **3 360 306 bytes** (~3.20 MiB), matching the expected
  EEGConformer-ONNX export footprint.
- Contract (per `scripts/export_braindecode_eegconformer.py` and the
  ONNX export parity audit): opset 17, input `[1, 22, 1000]` float32,
  outputs `embedding` `[1, 32]` and `logits` `[1, 4]`. The bridge
  registration in §1 reads exactly that contract (22 ch, 1000 samples,
  embeddingDim 32, embedding output name `"embedding"`).
- Parity: `cosine(PyTorch, ORT) > 0.999` per
  `docs/audits/onnx-export-parity-fix.md`.

The static contract is consistent end-to-end. Runtime opset
compatibility on browser `onnxruntime-web` is not exercised by this
audit (read-only scope).

## 4. Embedding dim contract

`registerBraindecodeEEGConformer({ artifact: "/models/eegconformer.onnx" })`
omits `embeddingDim`, so the default in
`src/lib/ai/models/registry.ts#registerBraindecodeEEGConformer` applies:
`embeddingDim ?? 32`. The wrapped `BraindecodeAdapter`/ONNX bridge
therefore advertises a 32-D embedding.

`src/lib/ai/validation/index.ts` validates dimension via
`validateEmbedding(vec, { expectedDim })`. There is no hard-coded
`32`; the dim is whatever the caller (or `embed()` facade) passes
as `expectedDim`. The `embed()` facade propagates the descriptor's
`embeddingDim`, so a 32-D output from EEGConformer flows through
validation cleanly. PCA's legacy dim is independent and is checked
against its own descriptor, so the fallback path is also clean.

Contract: **32-D, float32, L2-normalised**. Consistent across
registry, bridge, validation call site, and the production-readiness
baseline.

## 5. Unresolved items since last audit

From the 2026-06-17 production-readiness sign-off criteria:

1. **≥ 1 000 successful inferences in staging with < 0.5 % fallback
   rate** — not yet collected. No telemetry counter is wired against
   the prod id.
2. **P95 latency on target devices < 600 ms** — not measured. The
   benchmark harness exists (`src/lib/ai/benchmark/index.ts`) but no
   recorded run against the live `/models/eegconformer.onnx` artefact
   on representative hardware is in the repo.
3. **Cosine recall@10 vs PCA improves by ≥ 15 % on the BCI-IV-2a
   holdout** — not measured against the deployed artefact. Training
   reports validation acc 58.7 % / holdout 57.8 %, but
   embedding-quality recall@10 vs PCA is not in any audit since
   2026-06-17.
4. **No security findings on the artefact host** — N/A; artefact is
   served from the app's own `/models/` static path.

Additional gap surfaced by §2: `embedEEG()`'s `DEFAULT_PREFERRED`
still names the legacy `braindecode-eegnetv4-onnx` id, so the prod
EEGConformer is reachable only via explicit `preferredModelId`
overrides.

## 6. Updated maturity scores

| Score                     | Previous | Current  | Δ   |
| ------------------------- | -------- | -------- | --- |
| Overall Platform Maturity | 58 / 100 | 64 / 100 | +6  |
| Production Readiness      | 55 / 100 | 62 / 100 | +7  |
| AI Layer Readiness        | 60 / 100 | 72 / 100 | +12 |

Justification:

- **Overall +6**: The AI layer now ships an end-to-end production
  artefact (registered model + ONNX file in `public/models/` + parity
  ≥ 0.999 + falling-back chain). Other platform pillars (auth, vector
  store, UI) are unchanged, so the lift is bounded.
- **Production Readiness +7**: Registration, artefact, and validation
  contracts are coherent (§1–§4). What blocks a higher score is
  unchanged from 2026-06-17 sign-off: no staging inference volume,
  no measured P95 latency, no recall@10 evidence (§5), and the
  `DEFAULT_PREFERRED` mismatch in `embedEEG()` means default callers
  still hit PCA without an explicit override.
- **AI Layer Readiness +12**: The largest delta. The chain
  EEGConformer → generic ONNX → PCA is fully wired in code, the
  artefact is committed, dim/parity contracts agree, and unit tests
  cover registration plus the PCA-fallback edge. Remaining gap is the
  default-id wiring noted in §2 and the absence of a live
  benchmark/recall measurement against the deployed file.
