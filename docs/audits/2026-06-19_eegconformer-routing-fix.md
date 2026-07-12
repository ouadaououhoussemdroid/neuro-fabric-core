# EEGConformer Production Routing Fix

- Date: 2026-06-19
- Scope: production routing only. No model retraining, no weight changes,
  no ONNX regeneration.

## Problem

`embedEEG()` defaulted to the legacy id `braindecode-eegnetv4-onnx`, which
is **not** registered in the current registry. Default callers therefore
silently fell through to the PCA fallback instead of routing to the
production EEGConformer artefact at `/models/eegconformer.onnx`.

## Verified registered id

`src/lib/ai/models/registry.ts` calls
`registerBraindecodeEEGConformer({ artifact: "/models/eegconformer.onnx" })`
at module load. `registerBraindecodeEEGConformer` defers to
`registerBraindecodeONNX` with `id: opts.id ?? "braindecode-eegconformer-prod"`.

Confirmed registered id: **`braindecode-eegconformer-prod`**.

## Files modified

- `src/lib/ai/inference/embed-eeg.ts`
  - `DEFAULT_PREFERRED`: `"braindecode-eegnetv4-onnx"` → `"braindecode-eegconformer-prod"`

No other source files required changes. Remaining textual matches for
`braindecode-eegnetv4-onnx` are:

- Historical audit/roadmap docs under `docs/audits/` and `docs/roadmaps/`
  (intentionally preserved as historical record).
- `src/lib/ai/adapters/__tests__/braindecode-onnx-bridge.test.ts` — exercises
  the generic `registerBraindecodeONNX` path with its default id; unrelated
  to production routing and left intact.

## Final routing chain

Default `embedEEG()` call (no `preferredModelId` override):

```
braindecode-eegconformer-prod   (start; loads /models/eegconformer.onnx)
  → <opts.onnxModelId>          (only if caller provided and registered)
  → pca-legacy-v1               (always-available terminal fallback)
```

PCA is reached **only** when EEGConformer load or inference fails (no
WASM runtime, fetch error, validation failure). The `embed()` facade
still walks `fallbackChain` and reports `fellBack: true` with a reason.

## Confirmation

- `hasModel("braindecode-eegconformer-prod")` is true at module load
  because `registerBraindecodeEEGConformer(...)` runs at top level of
  `registry.ts` before any `embedEEG()` call.
- `embedEEG()` resolves `startId = preferred` when registered, so the
  inference engine instantiates the EEGConformer adapter first.
- Existing test `eegconformer-registration.test.ts` already verifies
  the PCA fallback path with `preferredModelId` override and continues
  to pass under the new default.

Default inference now starts with EEGConformer; PCA is reached only on
failure.
