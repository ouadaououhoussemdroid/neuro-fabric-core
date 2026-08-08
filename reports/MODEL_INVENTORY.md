# Neuro-Fabric Core Model Inventory

## Executive Summary

This document provides a comprehensive inventory of all AI models currently integrated, referenced, or planned for the Neuro-Fabric Core platform. Each model is cataloged with details about its purpose, current status, maturity level, inference capabilities, training status, ONNX availability, and production readiness.

The platform currently features **one production-ready model** (EEGConformer), several **registered stubs** for future models, and **planned models** awaiting external dependencies or development effort.

## Model Taxonomy

Models are categorized by their role in the AI foundation layer:

1. **Baseline Models:** Simple, deterministic fallbacks (e.g., PCA)
2. **Production Models:** Validated, deployed models serving primary use cases
3. **Registered Stubs:** Models registered in the registry but not yet functional
4. **Planned Models:** Models planned for future implementation
5. **Experimental Models:** Models used for research or ablation studies

## Detailed Model Inventory

### 1. Baseline Models

#### PCA Legacy Embedder (`pca-legacy-v1`)

- **Purpose:** Deterministic baseline embedding via Principal Component Analysis
- **Status:** Implemented and integrated
- **Maturity:** Prototype (fallback-only)
- **Inference:**
  - Browser-based JavaScript implementation
  - Deterministic, no external dependencies
  - Fixed dimensionality (configurable, default 32)
- **Training Status:** N/A (not learned; projection matrix computed on first use or pre-fitted)
- **ONNX Availability:** N/A (not an ONNX model)
- **Production Readiness:**
  - ✅ Deterministic and reliable
  - ❌ Limited expressive power (linear transformation only)
  - ✅ Zero latency after initialization
  - ✅ Serves as critical fallback for robustness
- **Evidence:**
  - `src/lib/ai/adapters/pca-adapter.ts`
  - Used as terminal fallback in embedding orchestration (`src/lib/ai/inference/embed-eeg.ts`)

### 2. Production Models

#### EEGConformer Production Model (`braindecode-eegconformer-prod`)

- **Purpose:** General-purpose EEG embedding and motor imagery classification
- **Status:** Live and verified in default routing
- **Maturity:** Research Platform (validated technically, awaiting empirical validation)
- **Inference:**
  - Browser-based via ONNX Runtime Web
  - Execution providers: WASM (primary), WebGPU/WebGL fallbacks
  - Input: 22-channel EEG @ 250 Hz, 1000-sample windows (4 seconds)
  - Outputs:
    - Embedding: 32-dimensional attention-pooled features
    - Logits: 4-class motor imagery (left hand, right hand, feet, tongue)
- **Training Status:**
  - ✅ Trained on BCI-IV-2a dataset via MOABB
  - ✅ Preprocessing: Bandpass (0.5-40 Hz), epoching, standardization
  - ✅ Optimizer: AdamW with weight decay
  - ✅ Validation: Cross-subject hold-out
  - 🔄 Empirical validation pending (BCI-IV-2a holdout evaluation)
- **ONNX Availability:**
  - ✅ Exported to ONNX (opset 17)
  - ✅ Verified PyTorch→ORT cosine similarity >0.999
  - ✅ Contains both embedding and logits outputs
  - 📍 Location: `public/models/eegconformer.onnx`
  - 🔐 Integrity: SHA-256 hash recorded in `public/models/manifest.json`
- **Production Readiness:**
  - ✅ Technically verified (end-to-end in browser)
  - ⚠️ Empirical validation pending (scientific validity)
  - ✅ Robust fallback to PCA
  - ✅ Model-ID tagging prevents vector contamination
  - ⏳ Awaiting BCI-IV-2a holdout evaluation for production claims
- **Evidence:**
  - `src/lib/ai/models/registry.ts` - Registration as `braindecode-eegconformer-prod`
  - `src/lib/ai/adapters/braindecode-onnx-bridge.ts` - ONNX bridge implementation
  - `public/models/eegconformer.onnx` - The ONNX artefact
  - `public/models/manifest.json` - SHA-256 hash and metadata
  - `training/` - Complete training pipeline
  - `docs/audits/2026-06-19_project_state_audit.md` - Live verification and inconclusive empirical evaluation

### 3. Registered Stubs (Not Functional)

#### EEGNetv4 Default (`braindecode-eegnetv4-default`)

- **Purpose:** Compact CNN for EEG classification/embedding (baseline comparison)
- **Status:** Registered but not functional (throws on load)
- **Maturity:** Prototype (stub)
- **Inference:**
  - Not currently functional (requires Pyodide bridge)
  - Planned input: EEG windows
  - Planned output: Embedding/logits (dimensions TBD)
- **Training Status:** Not trained
- **ONNX Availability:** Not exported
- **Production Readiness:**
  - ❌ Not functional (throws `Braindecode bridge unavailable`)
  - ⏳ Awaits Pyodice+PyTorch bridge or ONNX export
  - 📋 Registered for future ablation studies
- **Evidence:**
  - `src/lib/ai/models/registry.ts` - Registration block (lines 77-86)
  - `src/lib/ai/adapters/braindecode-adapter.ts` - EEGNetv4 specification in `BRAINDECODE_MODELS`

#### ShallowFBCSPNet Default (`braindecode-shallowfbcspnet-default`)

- **Purpose:** Shallow ConvNet for motor imagery (baseline comparison)
- **Status:** Registered but not functional (throws on load)
- **Maturity:** Prototype (stub)
- **Inference:**
  - Not currently functional (requires Pyodide bridge)
  - Planned input: EEG windows
  - Planned output: Embedding/logits
- **Training Status:** Not trained
- **ONNX Availability:** Not exported
- **Production Readiness:**
  - ❌ Not functional (throws `Braindecode bridge unavailable`)
  - ⏳ Awaits Pyodice+PyTorch bridge or ONNX export
  - 📋 Registered for future ablation studies
- **Evidence:**
  - `src/lib/ai/models/registry.ts` - Registration block (lines 87-96)
  - `src/lib/ai/adapters/braindecode-adapter.ts` - ShallowFBCSPNet specification in `BRAINDECODE_MODELS`

#### Deep4Net Default (`braindecode-deep4net-default`)

- **Purpose:** Deep ConvNet for EEG classification (baseline comparison)
- **Status:** Registered but not functional (throws on load)
- **Maturity:** Prototype (stub)
- **Inference:**
  - Not currently functional (requires Pyodide bridge)
  - Planned input: EEG windows
  - Planned output: Embedding/logits
- **Training Status:** Not trained
- **ONNX Availability:** Not exported
- **Production Readiness:**
  - ❌ Not functional (throws `Braindecode bridge unavailable`)
  - ⏳ Awaits Pyodice+PyTorch bridge or ONNX export
  - 📋 Registered for future ablation studies
- **Evidence:**
  - `src/lib/ai/models/registry.ts` - Registration block (lines 97-106)
  - `src/lib/ai/adapters/braindecode-adapter.ts` - Deep4Net specification in `BRAINDECODE_MODELS`

#### EEGPT Placeholder (`eegpt-placeholder`)

- **Purpose:** High-dimensional foundation model for EEG representation learning
- **Status:** Explicit stub (throws on all methods)
- **Maturity:** Scheduled (blocked on external dependencies)
- **Inference:**
  - Not implemented (throws `NotImplementedError`)
  - Planned: Server-side inference (512-D too large for browser WASM)
  - Alternative: Quantized WebGPU ONNX if <50 MB
- **Training Status:** Not trained
- **ONNX Availability:** Not available (blocked on checkpoint)
- **Production Readiness:**
  - ❌ Explicitly not implemented
  - 🚫 Blocked by: Lack of public, license-clear EEGPT checkpoint
  - 📋 Registered to show intent in model list
  - 🔓 Unblocking conditions: 1. Publicly distributable EEGPT checkpoint with clear license (BSD-3/MIT/Apache-2.0 preferred) 2. Verified ONNX export path (PyTorch→ORT cosine >0.999) 3. Runtime decision: Server-side (recommended) or quantized WebGPU ONNX
- **Evidence:**
  - `src/lib/ai/adapters/eegpt-adapter.ts` - Explicit stub implementation
  - `src/lib/ai/models/registry.ts` - Registration as `EEGPTAdapter` (line 54)
  - Header documentation details unblocking conditions

### 4. Planned Models

#### Cognitive Decoder v1 (Planned)

- **Purpose:** Predict cognitive states (attention, workload, arousal) from EEG
- **Status:** Planned (not yet registered)
- **Maturity:** Planned
- **Inference:**
  - Planned formats: Browser-based ONNX or TensorFlow.js
  - Input: Preprocessed EEG windows or features
  - Output: Continuous values for attention, workload, arousal (0-1 scale)
- **Training Status:** Not trained
- **ONNX Availability:** Planned export target
- **Production Readiness:**
  - ❌ Not implemented
  - 📋 Replacement for heuristic ratios currently in `src/lib/decoder/`
  - 🎯 Target: Train on public dataset with labeled cognitive states
  - 🔓 Unblocking conditions: 1. Labeled dataset with EEG and cognitive state annotations 2. Training pipeline for lightweight model (logistic regression → shallow NN) 3. ONNX/TF.js export for browser deployment
- **Evidence:**
  - `src/lib/decoder/` - Heuristic implementation (to be replaced)
  - `scripts/train_cognitive_decoder.py` - Placeholder training script
  - Roadmap documents indicating Month 3 objective

### 5. Experimental Models (For Ablation Studies)

#### Planned ONNX Variants of Braindecode Models

- **Purpose:** Enable fair comparison of architectures via common ONNX runtime
- **Status:** Planned (requires ONNX export)
- **Models:** EEGNetv4, ShallowFBCSPNet, Deep4Net, EEGConformer (additional variants
- **Inference:**
  - Browser-based via ONNX Runtime Web (same infrastructure as production model)
  - Shared input/output contract for fair comparison
- **Training Status:** Would be trained separately for each architecture
- **ONNX Availability:** Planned via `scripts/export_braindecode_eegconformer.py` (supports --architecture flag)
- **Production Readiness:**
  - ❌ Not yet exported
  - 🛠️ Enabled by T-015 extension to export script
  - 📋 Will allow ablation studies without changing inference infrastructure
  - ✅ Will reuse existing ONNX adapter and fallback chain
- **Evidence:**
  - `scripts/export_braindecode_eegconformer.py` - Supports `--architecture` flag (T-015)
  - `src/lib/ai/adapters/braindecode-onnx-bridge.ts` - Generic ONNX bridge
  - `src/lib/ai/models/registry.ts` - Registration helper `registerBraindecodeONNX`

## Model Registry Mechanics

### Registration Functions

All models are registered via the central registry in `src/lib/ai/models/registry.ts`:

1. `registerModel(factory: AdapterFactory): void` - Generic registration
2. `registerBraindecodeONNX(opts: ONNXBraindecodeBridgeOptions & {id?: string}): string` - Braindecode models via ONNX
3. `registerBraindecodeEEGConformer(opts: {artifact: string, ...}): string` - Specialized for production EEGConformer

### Model Lifecycle

1. **Registration:** Model factory added to registry (typically at module initialization)
2. **Resolution:** `createAdapter(id: string)` retrieves factory and creates instance
3. **Loading:** Adapter's `load()` method acquires resources (e.g., loads ONNX session)
4. **Inference:** `embed()` or `predict()` methods called on loaded adapter
5. **Unloading:** Adapter's `unload()` method releases resources (called during cleanup)

### Fallback Chain

The embedding orchestrator (`src/lib/ai/inference/embed-eeg.ts`) implements:

1. Try requested model ID (if specified)
2. Try default production model (`braindecode-eegconformer-prod`)
3. Try any registered ONNX model
4. Fall back to PCA legacy (`pca-legacy-v1`)

This ensures graceful degradation while allowing model experimentation.

## Recommendations for Model Expansion

### Immediate Term (0-3 months)

1. **Validate EEGConformer:** Complete BCI-IV-2a holdout evaluation to establish empirical baseline
2. **Export Alternative Architectures:** Use T-015 extended script to produce ONNX for EEGNetv4, ShallowFBCSPNet, Deep4Net
3. **Implement Pyodide Bridge:** Evaluate feasibility for running PyTorch models in browser (if ONNX not suitable for certain architectures)

### Medium Term (3-6 months)

1. **Develop Cognitive Detector:** Train and deploy first task-specific model (attention/workload/arousal)
2. **Evaluate EEGPT Alternatives:** Investigate other foundation models if EEGPT remains blocked
3. **Implement Model Versioning:** Enhance registry to support semantic versioning and A/B testing

### Long Term (6-12 months)

1. **Foundation Model Assessment:** Re-evaluate EEGPT or consider alternatives (BENDR, BIOT, LaBraM) if checkpoint becomes available
2. **Multi-Task Heads:** Develop infrastructure for task-specific heads on shared backbone
3. **Model Zoo Curation:** Establish criteria for model inclusion based on validation benchmarks

## Model Performance Characteristics

| Model                       | Embedding Dim     | Input Shape   | Compute Profile       | Fallback Safety       | Validation Status        |
| --------------------------- | ----------------- | ------------- | --------------------- | --------------------- | ------------------------ |
| PCA Legacy                  | Configurable (32) | Features (N)  | O(N*D) CPU            | N/A (baseline)        | Deterministic            |
| EEGConformer                | 32                | [1, 22, 1000] | WASM SIMD (~10-50ms)  | ✅ PCA                | Technical ✓, Empirical ? |
| EEGNetv4                    | 16                | [1, 22, 256]  | WASM SIMD (~5-20ms)   | ✅ Plato              | Not implemented          |
| ShallowFBCSPNet             | 40                | [1, 22, 1125] | WASM SIMD (~10-30ms)  | ✅ PCA                | Not implemented          |
| Deep4Net                    | 200               | [1, 22, 1125] | WASM SIMD (~20-100ms) | ✅ PCA                | Not implemented          |
| EEGPT (planned)             | 512               | [1, 22, 1024] | Server/WebGPU (~?)    | ⚠️ Server dependency  | Not implemented          |
| Cognitive Decoder (planned) | 3 (AWL)           | Features/TBD  | WASM/TF.js (~1-5ms)   | ✅ Heuristic fallback | Not implemented          |

## Conclusion

The Neuro-Fabric Core platform demonstrates a mature approach to model management through its adapter pattern and registry system. While currently featuring only one production-ready model (EEGConformer), the infrastructure is in place to rapidly expand the model zoo as validation and development efforts progress.

The critical path forward involves:

1. Validating the existing production model empirically
2. Expanding the model zoo with comparable alternatives for ablation studies
3. Developing the first task-specific model (cognitive decoder)
4. Establishing ongoing model evaluation and promotion processes

This foundation ensures that as new EEG models become available (whether from academic research or industry), they can be integrated, validated, and deployed with minimal friction, maintaining the platform's position at the forefront of EEG-based brain-computer interface technology.
