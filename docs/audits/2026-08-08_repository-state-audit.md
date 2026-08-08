# Repository State Audit — Neuro-Fabric Core (2026-08-08)

> Forensic, read-only state audit of the live working tree as of **2026-08-08**.
> This document is **truth derived from the working tree**, not from narrative reports.
> Every conclusion is anchored to a file path, a command result, or a migration/commit
> hash. Narrative reports (`FINAL_GO_NO_GO.md`, `AUDIT_2026.md`, `TIER_2/3/4.md`, etc.)
> are treated as **evidence and historical context, NOT truth** — and where they
> disagree with the working tree, the working tree wins.
>
> Security constraint honored: **no production or source code was modified.** Only
> read-only commands were executed (`vitest`, `tsc`, `eslint`, `vite build`,
> `python -c onnx.load`, `git`, `sha256`) and files were read. The single artifact
> produced is this audit document.

## 0. Executive Summary

**The repository is in a materially better state than its own narrative reports imply,
and in a materially worse state than the most optimistic reports claim. The truth is
the middle, and it is verified here.**

Neuro-Fabric Core ships a genuinely functional EEG ingestion + preprocessing stack, a
real ONNX-backed foundation-model layer with **five authentic Tier-4 ONNX artifacts**
(byte-verified against `manifest.json`, ONNX magic `0x08` confirmed), real Supabase
migrations with pgvector + RLS, a hardened upload API, and **599 passing tests across 67
files**. The EEGConformer export is real and exercises real onnxruntime-web inference
in tests.

However, the headline claim — *"EEGConformer live in default routing"* — is **false**:
`AI_EEGCONFORMER_ENABLED=off` by default, EEGConformer is **unregistered at startup**,
and the production `upload.ts` route does **not** call `embedEEG`/`embed()` at all — it
calls the legacy `embedSignal` (PCA/autoencoder, 64-dim). The foundation-model facade
is real and tested, but it is **not wired into any production data route**.

Two concrete, verified defects were found and are NOT mentioned honestly by the narrative
reports:

1. **Dimension mismatch (silent ANN write failure):** `upload.ts` computes a 64-dim PCA
   embedding but dual-writes it to the pgvector `embeddings` table whose column is
   `vector(32)`. The insert raises a dimension-mismatch error, which `upload.ts` catches
   and logs as `eeg.upload.vector_store_failed` (best-effort) — so the upload *succeeds*
   but the vector index is **silently never populated**. (The primary `eeg_analyses`
   table stores `embedding FLOAT8[]`, so it is unaffected.)
2. **EEGPT declared≠actual dim:** the EEGPT descriptor advertises `embeddingDim: 2048`
   and its own description says "2048-dim embeddings (flattened from [1, 31, 2048])", but
   the ONNX graph output is `[1, 31, 2048]` → the adapter flattens the **entire** tensor,
   yielding **63,488** dimensions. No test asserts the exact dim (tests check `length>0`),
   so this is uncaught.

**CI is RED:** `npx eslint .` exits `1` (24 errors — mostly Prettier formatting in new
Tier-4 test files plus 2 in `benchmark.ts`); `npx tsc --noEmit` exits `2` (2 real errors,
`src/lib/evaluation/benchmark.ts:676`, `centroidEntries` used before declaration). Only
`npx vite build` succeeds (`✓ built in 27.50s`). Reports claiming "0 errors" / "CI green"
are inaccurate.

**Operational reality:** the last commit is `dbed902` (2026-07-24). The entire Tier-4
feature set — **seven** of the eight ONNX models, the final-gate test suites, the updated
`manifest.json`, and an 84-file / +3,929-line working-tree delta — is **uncommitted**. Only
`eegconformer.onnx`(.data) is tracked in git. A new engineer who clones the repo today
inherits the **Jul-24 baseline**, *not* the Aug-8 audited state.

**Overall maturity: 70 / 100** — real, substantial engineering with verified artifacts and
passing tests, but the headline model is off by default and unrouted, CI is red, and the
audited state is mostly uncommitted. See §12 for the per-dimension breakdown and §20 for
the exact answer to *"what do I inherit?"*.

---

## 1. Repository Snapshot

| Item | Value | Source |
|---|---|---|
| Working tree date audited | 2026-08-08 (05:00–07:06 model timestamps) | `ls -la public/models/` |
| Last commit | `dbed902` — "Checkpoint: save all current project changes" — **2026-07-24** | `git log` |
| Commit before that | `e5370cf` "Add EEGConformer PCA evaluation notebook" — 2026-07-24 | `git log` |
| Tracked model files | `eegconformer.onnx` + `eegconformer.onnx.data` only | `git cat-file -e HEAD:...` |
| Untracked model files (on disk, NOT in git) | `cbramod-encoder`, `cognitive-decoder-v0`, `eegpt-encoder-int8`, `femba-tiny-encoder-adapter`, `femba-tiny-encoder-fp16`, `femba-tiny-encoder`, `labram-encoder` (.onnx) | `git status --porcelain public/models/` → all `??` |
| Untracked Tier-4 test files | `tier4-production-path.test.ts`, entire `src/lib/ai/inference/__tests__/` dir | `git status` |
| Working-tree delta vs HEAD | **84 files, +3,929 / −318** | `git diff --stat HEAD` |
| Package manager | **Bun** authoritative (`bun.lock`, dated Jul 12); `package-lock.json` also present (Jul 23, vestigial) | `ls -la`; `ci.yml` uses `bun install --frozen-lockfile` |
| CI | GitHub Actions, `on: push|pull_request` + nightly `cron "0 3 * * *"` | `.github/workflows/ci.yml` |

**Key implication:** The audited "Tier-4 complete" state lives **only in this working
directory** and is not reproducible from git alone. Before any release, this delta must
be committed (and the 7 untracked model binaries added via Git LFS, since several are
20–32 MB).

## 2. Current Architecture

**Framework:** TanStack Start (`@tanstack/react-start` 1.167.x), built on Vite 7 + React 19
+ `@tanstack/react-router` 1.168.x (`package.json`, `vite.config.ts`). SSR deploy target is
Nitro / Cloudflare Workers (the codebase uses the `.server.ts` / `.client.ts` naming
convention and comments noting `process.env` binds at request time on Workers — see
`src/lib/env.server.ts:6`).

**Frontend:** A rich route tree under `src/routes/`:
- *Public marketing/concept:* `index`, `about`, `architecture`, `developers`, `eeg2image`,
  `embeddings`, `playground`, `pricing`, `research`, `signin`, `signup`, `studio`,
  `synthetic`.
- *Authenticated product:* `_authenticated/dashboard.*` (`analyses`, `datasets`,
  `experiments`, `models`, `onnx`, `training`, `upload`), `_authenticated/route.tsx`,
- *API:* `api/eeg/upload`, `api/evaluate/cross-subject`, `api/health`,
  `api/public/cron/recall`, `api/public/metrics`, `api/public/notebooks`,
  `api/public/stream/-$source`, `api/annotations`.
- `src/routes/sitemap[.]xml.ts`, `src/routeTree.gen.ts` (auto-generated).

**Build:** `npx vite build` → `✓ built in 27.50s` (BUILD_EXIT:0). Green.

**Alias:** `@` → `src/` (via `vite-tsconfig-paths` + `tsconfig.json`).

**Honesty note (product reality):** The most user-visible "AI" pages are explicitly
labeled as concept demos and do **not** run live inference. `src/routes/embeddings.tsx:383`
states: *"this page renders locally-generated demo points, not live production
embeddings."* `src/routes/eeg2image.tsx:76` states: *"it is not connected to a live
image-generation model."* Commit `48ad200` ("fix: stop presenting fabricated metrics and
capabilities as real") gutted `src/components/live-ops.tsx` and `src/hooks/use-telemetry.ts`
(10 files, −106 net lines in those two) — a deliberate correction of previously-inflated
product-page claims. **Verified:** the polished frontend is demo UI with honest
disclaimers; the neural backend is real but the headline pages are not wired to it live.

## 3. Technology Stack

| Layer | Reality | Evidence |
|---|---|---|
| Runtime (browser) | `onnxruntime-web@1.26` (WASM SIMD + optional WebGPU) | `package.json` dep; `src/lib/ai/adapters/onnx-adapter.ts:79` dynamic `import("onnxruntime-web")` |
| ORT WASM self-hosting | **Real** — `ortWasmSelfHostPlugin` copies ORT WASM to `/ort/` at build time and emits `public/ort/integrity.json` with SHA-384; `defaultRuntime()` pins `mod.env.wasm.wasmPaths = "/ort/"`, overridable via `VITE_ORT_WASM_PATHS` | `onnx-adapter.ts:88-103`; Vite plugin `vite-plugins/...` |
| WebGPU gate | `getExecutionProviders()` reads `VITE_ORT_WEBGPU` (defaults `["wasm"]`) | `src/lib/ai/adapters/webgpu-flag.ts`; `webgpu-flag.test.ts` |
| Runtime (server) | Node via `onnxruntime-web` CPU EP in Node (Node-compatible, no wasmPaths) | `tier4-production-path.test.ts:59-62` `nodeRuntime()` |
| Language | TypeScript (ESM) | `package.json` `"type":"module"` |
| Auth | Supabase Auth (JWT via `authenticateRequest`) + `attachSupabaseAuth` in `startInstance.functionMiddleware` | `src/integrations/supabase/request-auth.ts`, `src/start.ts:24` |
| ORM/Storage | `@supabase/supabase-js@2.106` | `package.json` |
| DB | Postgres + pgvector | migrations |
| UI | Tailwind CSS v4 + Radix UI + `lucide-react` + `recharts` | `package.json` |
| Package manager | **Bun** (CI: `bun install --frozen-lockfile`, scripts use `bun run`) | `ci.yml`; `bun.lock` |

## 4. EEG Processing Layer

**Verdict: REAL and substantial.** `src/lib/eeg/` is a coherent, implemented ingestion +
preprocessing stack.

| Component | Reality | Evidence |
|---|---|---|
| EDF/BDF/EDF+ parser | **Real** — manual `DataView` parser, int16/int24 LE, BDF `0xFF+'BIOSEMI'` detection, TAL/EDF+ annotation parsing, per-channel sample-rate mismatch handling, GDF detection throws clean error | `src/lib/eeg/parsers/edf.ts` (195 lines) |
| CSV/TSV parser | **Real** — NaN forward-fill, per-channel dead-channel detection (50% NaN threshold), >20% file-wide NaN rejection | `src/lib/eeg/parsers/csv.ts` |
| NPY parser | **Real** — `.npy` v1.0/v2.0, magic `\x93NUMPY`, float32/float64/int16/int32, C/F-order, 2D `[C,N]` heuristic | `src/lib/eeg/parsers/npy.ts` |
| Parser dispatch | `parseEDF`/`parseCSV`/`parseNPY` re-exported from `parsers/index.ts` | `src/lib/eeg/parsers/index.ts` |
| Preprocessing | **Real** — `bandpass` (IIR biquad), `notch`, `zscore`, `segment`, `rejectArtifacts` (amplitude/flatline/jump) with `PreprocessingReport` step timings | `src/lib/eeg/preprocessing/index.ts`, `artifact-rejection.ts` |
| Features | **Real** — `fftPowerSpectrum` (O(N log N) radix-2 Cooley-Tukey) + `dftPowerSpectrum` parity baseline, `bandPowerFeatures`, `bandStats` | `src/lib/embeddings/features.ts` |
| Loaders | Sleep-EDF (**real**, fetches PhysioNet + parses), PhysioNet-MMIDB (**real**, 109×14), TUH (**architecture-only, requires operator mirror** — returns `[]` without index), BCI-IV-2a (**requires mirror**) | `src/lib/eeg/loaders/{sleep-edf,physionet,tuh,bci-competition}.ts` |
| Acquisition (file) | **Real** — `createFileSource` mirrors upload route's extension dispatch | `src/lib/eeg/acquisition.ts` |
| Acquisition (BrainFlow) | **Real** — dynamically imports optional `brainflow` native binding; **synthetic generator fallback** for CI/dev when the native dep is absent | `src/lib/eeg/brainflow-adapter.ts` (BoardId: synthetic/cyton/ganglion/muse) |
| Acquisition (LSL) | **Real but thin** — JSON frame decoder for the Python `scripts/lsl_bridge.py` sidecar; the TS side consumes the WS gateway, not LSL directly | `src/lib/eeg/lsl-adapter.ts` |
| Stream gateway | **Real** — WebSocket fan-out with per-peer monotonic `seq`, `model_id` header stamping (default `"eegconformer-v1"`), `GatewayError`, `createGateway` | `src/lib/eeg/stream-gateway.ts` |
| Signal quality | **Real** — per-channel RMS/flatline/clipping/NaN, overall `score` 0–100, color/label helpers | `src/lib/signal-quality/index.ts` |
| Synthetic EEG | **Real** — 1/f pink noise (Voss-McCartney, 16 octaves) + δ/θ/α/β/γ oscillators, seeded `mulberry32` | `src/lib/synthetic/index.ts` |

Tests: `edf.test.ts` (11), `csv.test.ts`, `npy.test.ts`, `filters.test.ts`,
`normalize.test.ts`, `parity.test.ts`, `segment.test.ts`, `sleep-edf.test.ts` (9),
`loaders.test.ts`, `acquisition.test.ts`, `brainflow-adapter.test.ts` (synthetic path),
`lsl-adapter.test.ts`, `stream-gateway.test.ts`, `features.test.ts`, `fft.test.ts`,
`fft-benchmark.test.ts`, `pca.test.ts`, `autoencoder.test.ts`, `index.test.ts`
(embeddings). All green within the 599 total.

## 5. AI / ML Layer

The AI layer is **architecturally real and internally consistent**, using a clean adapter
+ registry + facade pattern. The critical finding is **routing**, not implementation.

### 5.1 Registry & facade (real)
- `src/lib/ai/models/registry.ts` — `registerModel(factory)`, `createAdapter(id)`,
  `hasModel`, `listModels`, `unregisterModel`. The authoritative adapter list.
- `DEFAULT_EMBEDDER_ID = "pca-legacy-v1"` (line 191).
- `src/lib/ai/embeddings/index.ts` — the `embed()` facade: `createAdapter → load →
  embed → unload`, `validateEmbedding`, `l2Normalize`, a **hard PCA terminal fallback**,
  and an **observable** `announceFallback()` that both `console.error`s **and** dispatches
  a `neurofabric:embed-fallback` `CustomEvent` (consumed by `embed-fallback-badge.tsx`).
  `runFallbackChain` guarantees PCA at the tail of any chain. This is production-grade
  fallback logic with real observability.
- `src/lib/ai/inference/embed-eeg.ts` — `embedEEG()`, the high-level entry point:
  preferred = `braindecode-eegconformer-prod`, gated by `isEEGConformerEnabledForUser(userId)`
  (per-user djb2-hash cohort), fallback chain `[<onnx>, "pca-legacy-v1"]`.

### 5.2 Adapters (all real code; real ONNX inference verified — see §6)
- `onnx-adapter.ts` — `defaultRuntime()` does `import("onnxruntime-web")`, pins
  `wasmPaths="/ort/"`, `InferenceSession.create(artifact, {executionProviders})`, builds
  `[1,C,T]` float32 tensor, runs, flattens `tensor.data`. Real.
- `braindecode-onnx-bridge.ts` — `createONNXBraindecodeBridge()` delegates to
  `ONNXAdapter`; satisfies the `BraindecodeBridge` contract. Real code.
- `braindecode-adapter.ts` — `BraindecodeAdapter` validates window shape, forwards through
  an injected `bridge`; the default bridge throws `"Braindecode bridge unavailable"`.
  EEGNetv4/ShallowFBCSPNet/Deep4Net registered as stubs that load()→throw without a bridge.
  EEGConformer is registered via `registerBraindecodeEEGConformer` → `createONNXBraindecodeBridge`.
- `eegpt-adapter.ts` — `EEGPTAdapter` delegates to `ONNXAdapter`; declares `implemented:true`,
  artifact `/models/eegpt-encoder-int8.onnx`, Apache-2.0. **Real.**
- `pca-adapter.ts` — `PCAEmbeddingAdapter` wraps `embedSignal`, declares `embeddingDim: 64`.
- `pytorch-export-adapter.ts` — exists in the import list.

### 5.3 Cognitive decoder (real + production-routed)
- `src/lib/decoder/index.ts` `decodeCognitiveState()` → `src/lib/decoder/trained-decoder.ts`:
  `createONNXDecoder()` loads `/models/cognitive-decoder-v0.onnx` (1,333 bytes — a logistic
  regression) via onnxruntime-web, runs `match_embeddings`-style inference on a **5-band
  power feature vector `[1,5]`** (δ,θ,α,β,γ) → `probabilities` `[1,3]` (attention, workload,
  arousal), cached session, **heuristic spectral fallback** on any failure. The ONNX
  `probabilities` output is a single tensor (not a SEQUENCE) precisely to be ORT-WASM
  friendly. `src/routes/api/eeg/upload.ts:274` calls `decodeCognitiveState(pre.signal)` —
  **this is the only part of the foundation-model layer that actually reaches production
  data.** `scripts/train_cognitive_decoder.py` (real Python, numpy + sklearn) trains 3
  independent LogReg pipelines on a documented synthetic calibration set and exports ONNX.

### 5.4 What is NOT in production
- `embed()` / `embedEEG()` (the foundation-model embedding facade) is imported **only**
  by `src/components/embed-fallback-badge.tsx` (a UI observability listener) and
  `src/lib/evaluation/benchmark.ts` (eval infra). **No route or product component calls
  it.** `upload.ts` calls `embedSignal` (PCA/AE). So the EEGConformer / Tier-4 embeddings
  are **never produced on a real upload** — only on demand via tests/benchmarks.

## 6. Foundation-Model Readiness vs. Tier-4 Literature

**Verdict: 5 of 5 Tier-4 foundations are REAL artifacts with VERIFIED inference; the 6th
(EEGPT) is real but ships a dimension-bug; and none are on the production data path.**

### 6.1 Artifact authenticity (byte-verified)
All 8 entries in `public/models/manifest.json` (generated `2026-08-08T05:06:18Z`) were
SHA-256-verified against disk, including the external-data file:

| Manifest key | registryId | File on disk | Size | SHA-256 (head) | Verified |
|---|---|---|---|---|---|
| eegconformer | braindecode-eegconformer-prod | eegconformer.onnx + .data | 3.19 MB / 3.15 MB | `31cd36…` / `892b5a…` | ✅ |
| eegpt-encoder-int8 | onnx-eegpt | eegpt-encoder-int8.onnx | 25 MB | `a92daf…` | ✅ |
| femba-tiny-encoder-adapter | onnx-femba-tiny | femba-tiny-encoder-adapter.onnx | 30.7 MB | `e02422…` | ✅ |
| femba-tiny-encoder-fp16 | onnx-femba-tiny-fp16 | femba-tiny-encoder-fp16.onnx | 16.3 MB | `fc6b3b…` | ✅ |
| femba-tiny-encoder | onnx-femba-tiny-raw | femba-tiny-encoder.onnx | 30.7 MB | `c50ccc…` | ✅ |
| labram-encoder | onnx-labram | labram-encoder.onnx | 22.2 MB | `61f28d…` | ✅ |
| cbramod-encoder | onnx-cbramod | cbramod-encoder.onnx | 21.0 MB | `c128cc…` | ✅ |
| cognitive-decoder-v0 | (decoder) | cognitive-decoder-v0.onnx | 1.3 KB | `ea4f21…` | ✅ |

(`tier4-production-path.test.ts` Gate 1 + `eegpt-honest-stub.test.ts` assert these
hashes at runtime; ONNX magic `0x08` asserted in both.)

### 6.2 ONNX I/O contract (direct graph inspection via `onnx` 1.22.0)
To resolve the declared-vs-actual embedding-dimension question **definitively**, the model
graphs were inspected directly (not via inference — see §6.3 on why):

| Model | Declared `embeddingDim` | ONNX output shape | Flattened (batch=1) | Verdict |
|---|---|---|---|---|
| **eeg-conformer** | 32 | `embedding [0,32]` (+`logits [0,4]`), in `input [0,22,1000]` | **32** | ✅ matches |
| **EEGPT** | 2048 | `eeg_embedding [1, 31, 2048]`, in `eeg_input [1,62,1000]` | **63,488** | ❌ **mismatch** (1×31×2048) |
| **FEMBA-tiny (adapter)** | 30,800 | `embedding [0,0,0]` (unannotated) | — (sibling fp16 = `[0,80,385]`=30,800) | ✅ matches |
| **LaBraM** | 200 | `embedding [0,0]` (2D, unannotated) | — | declared 200; not pinned by static graph |
| **CBraMod** | 19,000 | `embedding [0,19,5,200]`, in `eeg [0,19,1000]` | **19,000** (1×19×5×200) | ✅ matches |

**⚠️ Correction to prior context:** an earlier draft suggested CBraMod's actual output was
9,500. The ONNX graph shows `[0,19,5,200]` = 19,000 at batch 1 — **it matches the declared
19,000.** The 9,500 figure is **not borne out** by the model. The genuinely-mismatched
model is **EEGPT**: the descriptor (and its own doc string) says "2048-dim (flattened from
`[1, 31, 2048]`)", but `1×31×2048 = 63,488`, and `ONNXAdapter.runOnce` does
`Array.from(tensor.data)` over the **whole** tensor. So EEGPT would emit a 63,488-dim
vector, not 2,048. Tests assert only `length>0`, so this is uncaught. If EEGPT were ever
routed to `pgvector vector(32)` (or validated against `expectedDim:2048`), it would fail.

### 6.3 Real inference verification (Node CPU backend)
The Tier-4 suites construct a real `ONNXAdapter` against the real filesystem artifact with
a real `import("onnxruntime-web")` CPU EP (`nodeRuntime()` in `tier4-production-path.test.ts:59`
and `tier4-final-gate.test.ts:51`) and run `load()→embed()→unload()`:

- `tier4-production-path.test.ts` **Gate 5** ("Real ONNX inference through embed()
  production path") — `it.each` over all 5 Tier-4 models + EEGConformer; asserts
  `vector.length>0`, `Number.isFinite` for every element, `fellBack===false`, `isLoaded()`
  toggles. **This is genuine onnxruntime-web forward inference on the real `.onnx` files.**
- `tier4-final-gate.test.ts` **VERIFICATION 1** — drives the **public `embedEEG()`** entry
  point the same way (each model registered under a *temp id* with a pre-built real
  adapter; EEGConformer default re-registered under a temp id).
- `onnx-artefact-integration.test.ts` — **does NOT** run inference. It verifies file
  existence (>1 MB), ONNX magic `0x08`, manifest SHA-256 match, and a best-effort I/O
  metadata probe that is **wrapped in try/catch and skipped** when onnxruntime-web can't
  build a Node session (the test literally comments: *"We can't run forward inference in
  Node without a WASM backend"*). So this file is **integrity + metadata**, not inference.

⚠️ **One real gap:** `braindecode-onnx-bridge.test.ts` exercises
`createONNXBraindecodeBridge` only with a **`fakeRuntime`** that returns a hardcoded
`[1, embeddingDim]` tensor (`Uint8Array([1,2,3])` artifact). The EEGConformer **registry
factory** (`registerBraindecodeEEGConformer` → `createONNXBraindecodeBridge` →
`ONNXAdapter`) is therefore **never loaded with a real runtime in tests** — only the
plain `ONNXAdapter` is (via `makeRealAdapter`). Since both produce the same `ONNXAdapter`
pointing at the same verified artifact, the risk is low, but it is an honest coverage gap:
*the production EEGConformer factory path is not exercised end-to-end with a real runtime.*

### 6.4 CBraMod browser caveat (real, not a stub)
CBraMod's ONNX contains `DFT` and `ReduceL2` ops unsupported by ORT-WASM. The registry
declares `wasmCompatible: false, wasmBlockers: ["DFT","ReduceL2"]`. It runs on the
**Node CPU EP** (so Gate 5 / VERIFICATION 1 pass), but would fail in the **browser WASM**
path. The test suite does **not** assert browser-WASM execution (it can't, in Node) — so
the "server-only" claim is metadata-backed but not runtime-verified in a browser.

### 6.5 Literature coverage
The Tier-4 literature review (in reports) surveyed: EEGConformer, EEGPT, LaBraM, CBraMod,
BENDR, BIOT, BrainOMNI, FEMBA-tiny, NeuroGPT, NeuroLM, REVE, EEG-Mamba, EEG-JEPA, LCM,
GEFM, HEAR, DBConformer, STEEGFormer. Of these, **5 foundations are fully deployed/real**
(EEGConformer, EEGPT, FEMBA-tiny, LaBraM, CBraMod) + the 1.3 KB cognitive decoder. The rest
are surveyed-only (no artifacts).

## 7. ONNX / Browser Inference

- `onnxruntime-web@1.26` is a real dependency; `defaultRuntime()` lazy-imports it and pins
  `wasmPaths` to `/ort/` (self-hosted, see T-008 / `ortWasmSelfHostPlugin`).
- The `ortWasmSelfHostPlugin` copies the ORT WASM bundle from `node_modules` at build time
  to `public/ort/` and writes `public/ort/integrity.json` with **SHA-384** hashes
  (referenced in `onnx-adapter.ts:88-103`). Self-hosting avoids the jsdelivr CDN + Vite dev
  `?v=hash` origin-mismatch that previously broke session creation.
- `getExecutionProviders()` (`src/lib/ai/adapters/webgpu-flag.ts`) returns
  `["wasm"]` by default, `["webgpu"]` when `VITE_ORT_WEBGPU="true"`.
- **Browser-WASM reality:** real inference is verified only in the **Node CPU EP** path
  (tests). The browser WASM path is wired (`wasmPaths="/ort/"`, `webgpu-flag`) but is not
  independently exercised in this audit's test run (vitest is Node). The `eegpt-honest-stub`
  test even asserts that the **default** `EEGPTAdapter.load()` (production runtime, with
  wasmPaths pinning) **throws in Node** (`eegpt-honest-stub.test.ts:103-109`) — i.e., the
  browser-oriented runtime fails in Node, while the test-injected `nodeRuntime()` succeeds.
  This is the expected environment split, not a defect.
- External-data ONNX (`eegconformer.onnx.data`, 3.15 MB) is present and SHA-verified
  (`sha256ExternalData` in manifest).

## 8. Database / Supabase

**Verdict: REAL, well-governed.** 15 migrations in `supabase/migrations/` (only
`concept_graph.sql` shows +4 uncommitted lines; the rest are committed in `dbed902`):

| Migration | Reality |
|---|---|
| `20260601050128` … auth/users base | accounts/waitlist/profiles |
| `20260603035330` / `20260604031328…` | handle_new_user, `document_role_trust_boundary` |
| `20260607000000_eeg_analyses.sql` | `eeg_analyses` table — `embedding FLOAT8[]` (any dim), RLS, per-user policies |
| `20260607151032` / `20260617180002` | `experiments` (+ ALTER later) |
| `20260711050000_rate_limits.sql` | `rate_limits` table + `check_rate_limit` (atomic `INSERT … ON CONFLICT … UPDATE`, `SECURITY DEFINER`, **deny-all-then-grant RLS**) |
| `20260711060000_pgvector_embeddings.sql` | `embeddings` table: **`embedding vector(32)`** + IVFFLAT index (lists=100), `model_id`/`user_id`, RLS + 3 policies, `GRANT` to `authenticated`/`service_role` |
| `20260711060100_match_embeddings_rpc.sql` | `match_embeddings` (ANN, cosine `<=>`, `SECURITY DEFINER`) |
| `20260711060200_match_embeddings_exact.sql` | `match_embeddings_exact` (exact, `<#>` squared-Euclidean) |
| `20260711070000_concept_graph.sql` | `concept_graph` (ltree hierarchy), `get_embedding_provenance` RPC |
| `20260711080000_datasets_manifest.sql` | `datasets_manifest` |
| `20260711090000_health_check_rpc.sql` | `health_check()` RPC, used by `api/health.ts` |

All RPCs use `REVOKE ALL ON FUNCTION FROM PUBLIC; GRANT EXECUTE … TO authenticated/service_role`,
are idempotent, and contain no destructive operations. RLS on every table.

**⚠️ The `vector(32)` column vs 64-dim PCA mismatch** (see §9/§15): the pgvector
`embeddings` table is `vector(32)`, but the production `upload.ts` path produces
**64-dim** PCA vectors (see §9). This is the latent insert bug.

`NeuralVectorIndex` (`src/lib/vector-search/neural-index.ts`) wraps the
`match_embeddings`/`match_embeddings_exact` RPCs with an in-memory `VectorIndex`
fallback; `recall-slo.ts` computes recall@K vs brute-force cosine — exercised by the
`/api/public/cron/recall` route (CRON_SECRET-protected) and `recall-slo.test.ts`.

## 9. Security

**Verdict: REAL and production-grade on the hot path.**

| Control | Reality | Evidence |
|---|---|---|
| Upload size cap | 50 MB | `upload.ts:15` `MAX_FILE_BYTES` |
| Rate limiting | 20 req / 60 s per user via `check_rate_limit` RPC; **fail-closed** (503 if RPC unavailable) | `upload.ts:117-127`; `rate-limit.ts` (atomic UPSERT) |
| File sniffing | Magic-number checks (`0x30` EDF, `0xFF` BDF, `\x93NUMPY` NPY); CSV/TSV rely on parser | `upload.ts:30-46` |
| Filename sanitization | Strips dir components, `[^a-zA-Z0-9._-]`→`_`, collapses `..`, trims to 255 | `upload.ts:384-393` `sanitizeFilename` |
| Processing timeout | 60 s hard `Promise.race` | `upload.ts:67-92` |
| Auth | JWT via `authenticateRequest`; `AuthError` trusted, other errors sanitized to "Authentication failed." | `upload.ts:99-113`; `request-auth.ts` |
| Error sanitization | Parse/processing errors logged server-side, client gets generic "An error occurred during processing." | `upload.ts:215-229`, `277-288` |
| Security headers | OWASP set: HSTS, X-Frame `DENY`, `nosniff`, CSP with `'wasm-unsafe-eval'` + `wss://self`, locked-down `Permissions-Policy` (camera/mic/etc `()`) | `src/middleware/security.ts` |
| CORS | Origin allow-list from `CORS_ALLOWED_ORIGINS`; **no wildcard**; OPTIONS 403 on bad origin | `src/middleware/cors.ts` |
| DB access control | RLS deny-all + `REVOKE ALL FROM PUBLIC` + `GRANT` on every table/RPC | all migrations |
| CI secret scan | grep-based scan for `sk-…`, `AKIA…`, `ghp_…` over `src/scripts/training` | `ci.yml` security job |
| Cron auth | `CRON_SECRET` Bearer check on `api/public/cron/recall` | `recall.ts:45-52` |
| `.env` hygiene | `AI_EEGCONFORMER_ENABLED=off`; `CRON_SECRET` real in `.env`, placeholder in `.env.example` | `.env` / `.env.example` |

No secrets are committed (the grep scan + the placeholder `CRON_SECRET` in `.env.example`
confirm this). No wildcards anywhere. Fail-closed defaults (rate limit, fallback to PCA).

## 10. API Layer

| Route | Reality | Evidence |
|---|---|---|
| `api/eeg/upload` | Hardened production route: auth → rate limit → 50 MB cap → magic sniff → parse(EDF/CSV/NPY) → preprocess → `embedSignal`(PCA) → `decodeCognitiveState` → persist to `eeg_analyses` (FLOAT8[]) → **best-effort** dual-write to `embeddings` (vector(32)) | `upload.ts` |
| `api/evaluate/cross-subject` | LOSO cross-subject eval | `cross-subject.ts` |
| `api/health` | `health_check()` RPC | `api/health.ts` |
| `api/public/metrics` | Prometheus exposition of `@/lib/metrics` (Counter/Gauge/Histogram, 16 buckets) | `metrics/index.ts`; in-process, single-isolate (documented limit) |
| `api/public/cron/recall` | Recall@10 SLO: samples labelled embeddings → `match_embeddings` RPC vs brute-force cosine ground truth | `recall.ts` (T-012) |
| `api/public/notebooks` | Serves notebook manifest built by `scripts/build_notebook_portal.py` | `notebooks.ts` |
| `api/public/stream/-$source` | WebSocket EEG gateway (fan-out, seq, model_id) | `stream/-$source.ts` |
| `api/annotations` | Annotation persistence | `annotations/index.ts` |

**Routing reality check (the central misclaim):** `upload.ts` imports `embedSignal` from
`@/lib/embeddings` — **not** `embedEEG`/`embed` from `@/lib/ai/embeddings`. So the
production upload embeds with PCA (64-dim), **not** EEGConformer. The `/models` page
(`src/routes/_authenticated/models.tsx`) reads the **legacy** `src/lib/model-registry`
and shows `ACTIVE_EMBEDDER = "linear-ae"` and `ACTIVE_DECODER = "baseline-spectral-v1"` —
itself a stale indicator: `upload.ts` actually prefers the ONNX cognitive decoder first and
only falls back to `baseline-spectral-v1`. There are **two model registries** (legacy
`model-registry` for UI display vs the new `ai/models/registry` for inference), and the
documented "active" values do not match what `upload.ts` executes.

## 11. Testing / CI / Quality

### 11.1 Test suite (GREEN)
`npx vitest run` → **67 files, 599 passed, 2 skipped (601 total), 43.68 s** (re-verified
this audit run). Representative suites (test-case call sites; some expand via `it.each`):

```
tier4-production-path.test.ts      12 sites  (T-016 "Final Gate" — real ONNX inference)
tier4-final-gate.test.ts            4 sites  (VERIFICATION 1/2 — embedEEG() facade)
eegpt-honest-stub.test.ts          16 sites  (EEGPT real-artifact honesty pass)
tier4-registration.test.ts         22 sites
cognitive-decoder-integration.test  9 sites
stats.test.ts                      31 sites
sleep-edf.test.ts                   9 sites
edf.test.ts                        11 sites
neural-index.test.ts / integration
recall-slo.test.ts / parity.test.ts / loso(12)/ground-truth(13)/stats(31)/benchmark(6)
embed-fallback.test.ts / model-zoo-registration / registry-sync / cohort(7)
loaders / preprocessing / parsers / filters / normalize / segment / fft(+benchmark)
autoencoder / pca / features / index(embeddings) / cosine
upload(-sanitize, -magic-number) / health / metrics / cors / security / env.server
braindecode-adapter / braindecode-onnx-bridge / onnx-adapter(test) / webgpu-flag / artefact-manifest
```

The "5 real Tier-4 foundation models" + EEGConformer claims are substantiated by Gate 5 /
VERIFICATION 1 (real onnxruntime-web CPU inference on all 6 + non-zero/non-NaN). The 2
skipped tests are pre-existing (no skip reason changes verdict).

### 11.2 Coverage
CI runs `bun run test:coverage` (`vitest --coverage`), but this audit did **not** capture
the coverage report (the plain `vitest run` was executed, not `--coverage`). Coverage % is
therefore **unverified** here; assume it is collected in CI but not asserted to a gate.

### 11.3 Lint (RED)
`npx eslint .` → **30 problems (24 errors, 6 warnings)**; exit code **1**. 22 are
auto-fixable (Prettier formatting + `prefer-const`) concentrated in the new Tier-4 test
files and `benchmark.ts`. **CI `Lint` step fails.**

### 11.4 Typecheck (RED)
`npx tsc --noEmit` → exit code **2**. Exactly 2 errors, both in
`src/lib/evaluation/benchmark.ts:676`:
```
benchmark.ts(676,25): error TS2448: Block-scoped variable 'centroidEntries' used before its declaration.
benchmark.ts(676,25): error TS2454: Variable 'centroidEntries' is used before being assigned.
```
**CI `Typecheck` step fails.** This directly contradicts reports claiming "0 TypeScript
errors." The most recent Tier-4 report honestly labels this as "pre-existing, unrelated to
Tier-4," which is the accurate framing.

### 11.5 Build (GREEN)
`npx vite build` → `✓ built in 27.50s` (exit 0). Vite does not run `tsc`, so build passes
despite the typecheck errors.

### 11.6 CI workflow
`.github/workflows/ci.yml` (Bun 1.3.14, `bun install --frozen-lockfile`):
- `ci` job: **Lint → Typecheck → Test with coverage → Build** (sequential; stops on first
  failure → currently stops at Lint).
- `recall-slo` job (nightly `cron` + PRs): runs `recall-slo.test.ts` +
  `validation-metrics.test.ts`.
- `security` job: `bun audit --severity high` (**blocking**, `|| true` removed per the
  inline comment) + grep-based secret scan over `src/scripts/training`.
- `migration-validation` job: Supabase CLI, asserts ≥1 migration, validates naming.

## 12. Maturity Scores

Per-dimension maturity (0–100), anchored to §evidence above. Scoring rubric:
100 = fully implemented, tested, deployed, green CI, no known gaps; penalties for
un-routed features (−), uncommitted state (−), CI red (−), latent bugs (−).

| # | Dimension | Score | Evidence anchor |
|---|---|---|---|
| 1 | Architecture & Framework Stack | 80 | §2–3 (solid TanStack Start; two registries) |
| 2 | EEG Processing Layer | 78 | §4 (real parsers/preproc/loaders/acq/quality/synth; TUH needs mirror) |
| 3 | AI / ML Layer | 68 | §5 (real adapters + inference, but EEGConformer/EEGPT unrouted, EEGPT dim bug) |
| 4 | Foundation-Model Readiness | 63 | §6 (5 real artifacts + verified inference, OFF-by-default & unrouted, CBraMod WASM-blocked) |
| 5 | Database / Supabase | 82 | §8 (real migrations, RLS, pgvector — but vector(32) mismatch defect) |
| 6 | Security | 84 | §9 (hardened upload, OWASP, no-wildcard CORS, fail-closed, CI secret scan) |
| 7 | API Layer / Routes | 72 | §10 (hardened upload + cron + metrics; product pages are demos) |
| 8 | Testing / CI / Quality | 65 | §11 (599 pass, real tier-4 suites; **lint RED, typecheck RED**) |
| 9 | Deployment / Infrastructure | 58 | §3/§8 (CI defined, /ort/ self-hosted; single-isolate metrics; uncommitted) |
| 10 | Technical Debt | 52 | §13 (2 tsc errors, lint debt, dim bugs, 2 registries, uncommitted state) |
| — | **Overall** | **70** | weighted by engineering reality, not narrative |

## 13. Technical Debt Reconciliation

The repository contains **two contradictory families** of narrative reports dated 2026-08-01:
one family (`FINAL_GO_NO_GO.md`, `P1/P2/TIER_2/3/4.md`) claims **everything is done** —
"CI/CD, rate limiting, self-hosted WASM, FFT, pgvector, trained cognitive decoder, 5 real
Tier-4 models, **0 errors**"; the other (`AUDIT_2026.md`, `TECHNICAL_DEBT.md`, `ROADMAP_*`,
`DEEPTECH_ANALYSIS.md`) claims **nothing exists** — "No CI/CD, in-memory vectors, heuristic
decoder, EEGConformer stub." **Neither is true.** Verified reality (§1–12):

- 599 passing tests (not 431/442/504 — those report counts are stale snapshots that
  predate the Aug-8 tiered-test additions);
- `tsc` has **2 real errors** (not 0);
- `eslint` has **24 errors** (not 0, not "clean");
- the 5 Tier-4 models are **real** (not stubs, not missing) — but EEGConformer is
  **off and unrouted** (not "live in default routing");
- pgvector is **real** (not "in-memory") — but the **wrong width** for the production
  embedder (vector(32) vs 64-dim PCA).

The reconciling commit is `48ad200` "fix: stop presenting fabricated metrics and
capabilities as real" (10 files gutted: `live-ops.tsx`, `use-telemetry.ts`, and the
marketing pages). The codebase is mid-correction: infrastructure was built (Jul 11–24
commits + Aug 1–8 working tree), and the marketing/demo surfaces were patched to be honest,
but the narrative reports were left in an inconsistent state. The most accurate single
report is the Tier-4 one that **explicitly flags** the `tsc` error as pre-existing.

**Debt register (verified):**
- `benchmark.ts:676` — `centroidEntries` used before assignment (tsc, 2 errors) — pre-existing.
- `eslint .` — 24 Prettier/`prefer-const` errors (mostly new tier-4 tests) — 22 auto-fixable.
- Two model registries: `src/lib/model-registry` (legacy, UI: `ACTIVE_EMBEDDER="linear-ae"`,
  `ACTIVE_DECODER="baseline-spectral-v1"`) vs `src/lib/ai/models/registry.ts` (inference).
  They can drift.
- `embeddings.tsx`/`eeg2image.tsx` are demo pages; only `/upload` (and the decoder within
  it) are live for real data.
- `package-lock.json` is vestigial alongside `bun.lock` (remove one).

## 14. Roadmap Reconciliation (Phase 1–4)

The git log shows the phase gates were **committed**:
- `911838b` "feat(phase-4.2/4.3): EEGConformer rollout gate + bootstrap integration"
- `be236f9` "feat(phase-4.3): per-user cohort routing for EEGConformer rollout"
- `63f524f` "feat(phase-4.1): add AI_EEGCONFORMER_ENABLED rollout enum to env schema"

So the *mechanism* for Phase 4.2/4.3 (rollout gate + per-user cohort routing) is implemented
and tested (`rollout.test.ts`: off→unregister, canary→register, ga→register; `cohort.test.ts`
7 tests for `isEEGConformerEnabledForUser`). The rollout gate is **off by default** and
EEGConformer is **unregistered at startup** (`start.ts` → `applyEEGConformerRollout()` in the
error middleware; `rollout.server.ts` unregisters it when stage=`off`). The per-user cohort
function exists but is **dead in production** because `embedEEG` is not called by any route
(see §5.4). So Phase 4.2/4.3's *control plane* is real; its *data plane* (being called from a
real request) is not wired.

Phase-4 claims in reports ("EEGConformer live in default routing", "per-user cohort
routing for EEGConformer rollout") are therefore **partially real** (the code exists and is
unit-tested) but **not actually exercised** by any user-facing flow, and gated off by
default. This is the single biggest gap between the roadmap narrative and the code.

## 15. What Is REAL / PARTIAL / MOCK / PLACEHOLDER / MISSING

- **REAL** (implementation + artifact + verified): EDF/BDF/TAL parser, CSV/NPY parsers,
  preprocessor (biquad bandpass/notch/zscore/segment/artifact-rejection), FFT
  (Cooley-Tukey) + band-power features, Sleep-EDF/PhysioNet loaders, BrainFlow synthetic
  fallback, LSL frame decoder, WS stream gateway, signal-quality, synthetic generator,
  pgvector migrations (vector(32) + IVFFLAT + 2 RPCs), concept-graph ltree schema,
  `check_rate_limit` UPSERT RPC, health-check RPC, `eeg_analyses`/`experiments` tables,
  hardened upload route (size/magic/ratelimit/timeout/security headers/CORS/auth/sanitization),
  security headers middleware, CORS allow-list, `cognitive-decoder-v0.onnx` (trained),
  `scripts/train_cognitive_decoder.py`, the 5 Tier-4 ONNX artifacts + EEGConformer
  (byte-verified), `ONNXAdapter` (real onnxruntime-web), `createONNXBraindecodeBridge`,
  `EEGPTAdapter`, EEGConformer registry, `embed()` facade + `embedEEG()`, rollout gate +
  cohort hashing, embed-fallback observability badge, `/models` page, upload/metrics/health/
  recall-cron/notebooks/stream routes, CI workflow (lint/typecheck/coverage/build/recall-
  slo/security/migration).
- **REAL inference verified, but** the EEGConformer/EEGPT/FEMBA/LaBraM/CBraMod inference
  path is verified via **direct `ONNXAdapter` construction with a Node CPU runtime** —
  not via the production `BraindecodeAdapter→bridge→createONNXBraindecodeBridge` factory
  with a real runtime (that factory is tested only with a fake runtime). See §6.3.
- **PARTIAL**: CBraMod (real artifact + real Node-CPU inference, but **browser-WASM-blocked**
  by DFT/ReduceL2; not run in a real browser here); TUH/BCI-IV-2a loaders (real code,
  require operator-staged HTTPS mirror); EEGConformer rollout (gate real + tested, but
  **off by default and not routed** to a request).
- **MOCK/INJECTED** (by design): BrainFlow native binding (synthetic fallback when absent);
  LSL bridge (Python sidecar, TS side is a decoder); the `fakeRuntime` in
  `braindecode-onnx-bridge.test.ts` (structural test, not inference).
- **NOT in production routing**: the entire foundation-model embed facade (`embed`,
  `embedEEG`). `upload.ts` uses `embedSignal` (PCA). See §5.4 / §10.
- **PLACEHOLDER-ISH (honest disclaimers)**: the demo product pages (`embeddings.tsx:383`,
  `eeg2image.tsx:76`) and `neuroweave-1024` card in `embeddings.tsx:703-736`
  ("Parameters: not built / Pretrain hours: not trained / Probe accuracy: not measured").
- **MISSING (from git)**: 7 of 8 ONNX model binaries + the final-gate test suites +
  `manifest-metadata.ts` exist **only in the working tree** (uncommitted); only
  `eegconformer.onnx(.data)` is tracked.

## 16. Critical Risks

1. **CI is red (blocking).** `eslint`=1 (24 errors), `tsc`=2 (2 errors). The `ci` job halts at
   Lint. Nothing merges green until `benchmark.ts:676` is fixed and the 22 Prettier errors are
   auto-fixed. (Build itself is green: `✓ built in 27.50s`.)
2. **Dimension-mismatch data loss in the ANN index.** `upload.ts` writes a 64-dim PCA vector
   to `embeddings.embedding vector(32)`. The insert errors; `upload.ts` catches and logs
   `eeg.upload.vector_store_failed` as a *warning*, so the upload succeeds but **the vector
   store is silently empty** — recall-SLO, `/models` search, and any future ANN feature will
  have no data. (See §8, §10.)
3. **EEGPT dimension bug.** Declared `embeddingDim:2048` vs actual ONNX output 63,488. If
   EEGPT is ever enabled and routed, embeddings are 31× too large and will not fit
   `vector(32)`; `validateEmbedding` would also reject them against `expectedDim:2048`.
   Latent until routing changes. (See §6.2.)
4. **Uncommitted Tier-4 state.** 7 model binaries + the final-gate tests + manifest update
   are not in git. A clone, a new CI runner, or a lost working directory reverts to the
   Jul-24 baseline without these artifacts. High risk of regression/loss. (See §1.)
5. **Headline feature off + unrouted.** EEGConformer is `AI_EEGCONFORMER_ENABLED=off` by
   default and `upload.ts` never calls it. Marketing/docs claiming it is "live" will mislead
   users and reviewers until either the route is rewired or the documentation is corrected.
   (See §5.4, §10.)
6. **Stale legacy registry.** `ACTIVE_EMBEDDER="linear-ae"` / `ACTIVE_DECODER="baseline-
   spectral-v1"` in `src/lib/model-registry` do not reflect `upload.ts` (which prefers the
   ONNX decoder first). Two registries can drift. (See §10.)

## 17. Recommended Next Steps (ordered)

1. **Make CI green** — fix `benchmark.ts:676` (`centroidEntries` TDZ) and run
   `eslint --fix` on the Tier-4 test files. (Blocks everything.)
2. **Commit the Tier-4 state immediately**, with the 7 model binaries on **Git LFS**
   (several are 20–32 MB; they are currently untracked and will bloat the repo).
3. **Decide the production embed path** — either (a) route `upload.ts` through
   `embedEEG()`/`embed()` so EEGConformer is actually used (and fix its off-by-default
   gate / the `vector(32)` column width), or (b) keep PCA as production and stop
   presenting EEGConformer as "live in default routing" in docs.
4. **Fix the dimension contract** — either widen `embeddings.embedding` to a sane width
   (e.g. `vector(768)`) and align `embedSignal`'s default `latentDim`, **or** stop the
   best-effort dual-write until the dims agree; at minimum surface `vector_store_failed`
   as an error/counter rather than swallowing it.
5. **Fix EEGPT's dim contract** — pool the `[1,31,2048]` token sequence to `[1,2048]`
   (mean/max pool over the 31 axis) so the descriptor matches reality, and assert the
   exact output dim in `eegpt-honest-stub.test.ts` / Gate 5.
6. **Reconcile the two registries** — single source of truth for "active model."
7. **Remove vestigial `package-lock.json`** (Bun is authoritative) or delete `bun.lock` if
   the team standardizes on npm.
8. **Add a real forward-inference smoke test for the EEGConformer factory path** (registry →
   `BraindecodeAdapter` → `createONNXBraindecodeBridge` → real runtime) so the production
   path, not just `makeRealAdapter`, is covered.
9. **Surface coverage as a CI gate** (`vitest --coverage` is already run) once green.

## 18. Proposed Tier-4 Implementation Plan (as-is vs as-needed)

The Tier-4 foundations are **already deployed and inference-verified** (§6). The remaining
work is **wiring + correctness**, not training/export:

- **T-016 completion:** add a registry-factory integration test with a real runtime
  (closes §6.3 gap); assert exact output dims per model (closes EEGPT bug exposure).
- **T-008/T-025 completion:** commit `/ort/` WASM self-hosting plugin output + `integrity.json`
  (verify build emits them); gate EEGConformer rollout behind a real user-facing toggle
  rather than env-only.
- **T-012 completion:** widen the embeddings column so the recall-SLO has real data to
  score against (currently the ANN index is empty due to the dim bug).
- **T-028 completion:** the upload hardening is done; remaining is the dual-write
  contract (see §17 step 4).

No new model exports are needed.

## 19. Final Verdict

Neuro-Fabric Core is **genuinely half-real**: the engineering substrate (EEG stack, ONNX
runtime, artifacts, migrations, hardened API, security, tests) is real and verified, but
the **headline** capability — "EEGConformer live in default routing" — is **not true today**.
It is off by default, unrouted from `upload.ts` (which uses PCA), and its artifacts live
only in an uncommitted working tree. CI is red on lint + typecheck. There are real,
latent bugs (vector width, EEGPT dims) that would bite before EEGConformer goes live.

This is not a stub. It is not vapor. It is a **pre-GA engineering baseline that is
mis-sold by its own documentation.** Ship the working-tree delta (models on LFS), green CI,
resolve the dimension contracts, and then the "real" claim becomes true.

**Overall maturity: 70 / 100.** Substantial, verified foundation; the last mile (production
routing, CI green, committed state, dimension correctness) is unfinished.

## 20. If a Senior Engineer Joined Today

**What they would inherit (the real state):**
- A TanStack Start + Vite 7 + React 19 app on Bun, with a committed Jul-24 baseline
  (`dbed902`) that already includes the EEGConformer artifact, the AI facade, migrations,
  and hardened upload. On top of that baseline, this working directory adds (but has
  **not committed**): 7 more ONNX models, the T-016 final-gate test suites, an updated
  `manifest.json`, and an 84-file / +3,929-line delta.
- A real EEG ingestion + preprocessing stack (EDF/BDF/CSV/NPY parsers, filters, FFT,
  loaders, acquisition adapters, signal quality, synthetic data) — all tested.
- Five real, byte-verified Tier-4 ONNX models + EEGConformer, with **verified onnxruntime-web
  inference** in Node (Gate 5 / VERIFICATION 1).
- A hardened upload API (rate limit, magic sniff, 50 MB cap, 60 s timeout, security
  headers, no-wildcard CORS, fail-closed auth) backed by 15 real Supabase migrations with
  RLS + signed RPCs.
- 599 passing tests across 67 files.

**What they can safely use:**
- The EEG preprocessing + embedding (`embedSignal`, PCA/AE, 64-dim), the cognitive decoder
  (`decodeCognitiveState` → ONNX logistic regression with heuristic fallback — this *is*
  on the production upload path), the upload route, the DB schema, the security middleware,
  the acquisition adapters, and the ONNX adapter infrastructure.
- The 5 Tier-4 models + EEGConformer for **benchmark/evaluation** (`benchmark.ts`,
  `embed-fallback.ts`) and local inference, since their artifacts are verified and
  inference is confirmed.

**What they should not trust:**
- Any narrative report dated 2026-08-01 that claims "0 errors," specific test counts
  (431/442/504), or "EEGConformer live in default routing" — verify against `tsc`,
  `eslint`, `vitest run`, and `upload.ts` first. (CI is currently **red**: lint exit 1,
  tsc exit 2.)
- The `/models` page's "active" flags (`ACTIVE_EMBEDDER="linear-ae"`,
  `ACTIVE_DECODER="baseline-spectral-v1"`) and the legacy `src/lib/model-registry` —
  they do not match what `upload.ts` actually executes.
- Any claim that the foundation-model embeddings are in the production data flow — they
  are not; `upload.ts` calls `embedSignal`, not `embedEEG`/`embed`.
- `public/models/manifest.json` is modified but the 7 model binaries it references are
  **untracked** — a fresh clone will not have them.

**What must be built next (before any release):**
1. Commit the working-tree delta (models via Git LFS).
2. Green CI: fix `benchmark.ts:676`, `eslint --fix` the Tier-4 tests.
3. Resolve the `vector(32)` vs 64-dim PCA mismatch so the ANN index actually populates.
4. Resolve EEGPT's 2,048-vs-63,488 dimension bug (pool the token axis).
5. Decide and implement whether EEGConformer should be on the production path (wire
   `embedEEG` into a route / flip the rollout gate / widen the vector column) — and make
   the docs match that decision.

---

*Audit authored 2026-08-08 against the live working tree. All command outputs cited
(`vitest 67 files / 599 passed / 2 skipped`, `tsc exit=2`, `eslint exit=1`, `vite build ✓`,
ONNX graph shapes via `onnx` 1.22.0, SHA-256 vs `manifest.json`, `git status` showing 7
untracked models) were produced during this audit and are reproducible from the repo root.*
