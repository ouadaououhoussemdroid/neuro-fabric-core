# Blueprint Preparation: Next Development Phase

> **⚠️ Historical document — superseded.** Retained as a baseline for traceability. The current project state is documented in `docs/audits/2026-06-19_project_state_audit.md`, and the active task catalogue is `docs/roadmaps/2026-06-19_open_source_execution_blueprint.md`.

**Planning Horizon:** 6-12 months  
**Goal:** Transform prototype into production neurotechnology platform  
**Date:** 2026-06-06

---

## PHASE 1: FOUNDATION (Months 1-2)

### 1.1 Data Persistence Layer

**Why:** Current code computes embeddings but discards them. Users cannot retrieve past analyses.

**What to Build:**

- Supabase PostgreSQL schema for:
  - `users` (identity)
  - `eeg_files` (uploaded files + metadata)
  - `eeg_analyses` (computed embeddings + decoder outputs)
  - `analysis_logs` (preprocessing steps + timings)

**Expected Impact:**

- ✅ Users can retrieve past analyses
- ✅ Foundational for audit trails
- ✅ Enables experiment tracking
- ✅ Required for scientific reproducibility

**Implementation Priority:** HIGHEST (blocks everything else)

**SQL Schema Example:**

```sql
CREATE TABLE eeg_analyses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id),
  file_name TEXT NOT NULL,
  file_size_bytes INT NOT NULL,
  sample_rate INT NOT NULL,
  num_channels INT NOT NULL,
  num_samples INT NOT NULL,

  -- Results
  embedding FLOAT8[] NOT NULL,
  embedding_dimensions INT NOT NULL,
  embedding_model TEXT NOT NULL,  -- 'pca', 'linear-ae', 'raw-bandpower'

  -- Cognitive metrics
  attention FLOAT8 NOT NULL,
  workload FLOAT8 NOT NULL,
  arousal FLOAT8 NOT NULL,

  -- Preprocessing
  bandpass_low FLOAT8,
  bandpass_high FLOAT8,
  notch_frequency INT,

  -- Metadata
  created_at TIMESTAMPTZ DEFAULT now(),
  processing_time_ms INT NOT NULL,

  UNIQUE(user_id, file_name, created_at)
);

CREATE TABLE preprocessing_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id UUID NOT NULL REFERENCES eeg_analyses(id) ON DELETE CASCADE,
  step_index INT NOT NULL,
  step_name TEXT NOT NULL,
  parameters JSONB,
  duration_ms FLOAT8,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

---

### 1.2 Authentication Integration

**Why:** API endpoint currently accessible without authentication. Security vulnerability.

**What to Build:**

- JWT validation middleware
- User context injection into request handlers
- Row-level security (RLS) in Supabase

**Expected Impact:**

- ✅ Only authenticated users can call API
- ✅ Users see only their own analyses
- ✅ Audit trail shows who performed which analyses
- ✅ Foundation for rate limiting per user

**Implementation:**

```typescript
// Add middleware to src/start.ts
const authMiddleware = createMiddleware().server(async ({ next, request }) => {
  const authHeader = request.headers.get("authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    throw new Error("Missing authorization header");
  }

  const token = authHeader.slice(7);
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser(token);

  if (error || !user) {
    throw new Error("Invalid token");
  }

  // Inject user into context
  return next({ user });
});
```

**Estimated Effort:** 3-4 days

---

### 1.3 Security Hardening

**Why:** No file size limits, no rate limiting, silent data corruption.

**What to Build:**

- 10MB file size limit
- Rate limiting (10 requests/minute per user)
- Fix NaN→0 silent conversion
- Add comprehensive error logging

**Expected Impact:**

- ✅ Prevent DoS attacks
- ✅ Detect data quality issues
- ✅ Easier debugging in production
- ✅ Audit trail for compliance

**Estimated Effort:** 2-3 days

---

## PHASE 2: SCALABILITY (Months 2-4)

### 2.1 Model Versioning & Registry

**Why:** When ML models are trained (future phase), need to track versions and compare performance.

**What to Build:**

- `model_versions` table tracking:
  - Model ID, version, creation timestamp
  - Performance metrics (AUC, F1, etc.)
  - Training dataset snapshot
  - Hyperparameters
  - Status (draft/validated/production)

- Model registry API:
  - `POST /api/models/register` — save new model
  - `GET /api/models/{id}/versions` — list versions
  - `POST /api/models/{id}/promote` — promote to production

**Expected Impact:**

- ✅ Can compare model performance over time
- ✅ Can rollback to previous models if needed
- ✅ Can A/B test different models
- ✅ Scientific reproducibility (can cite model version)

**Supabase Schema:**

```sql
CREATE TABLE model_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name TEXT NOT NULL,
  version_semver TEXT NOT NULL,  -- e.g., "1.0.0"
  model_type TEXT NOT NULL,  -- 'embedding-pca', 'embedding-ae', 'decoder-baseline'

  -- Training metadata
  training_dataset TEXT,
  n_training_samples INT,
  hyperparameters JSONB,

  -- Performance metrics
  validation_auc FLOAT8,
  validation_f1 FLOAT8,
  test_auc FLOAT8,
  test_f1 FLOAT8,

  -- Status
  status TEXT DEFAULT 'draft',  -- draft | validated | production
  promoted_by_user UUID REFERENCES auth.users(id),
  promoted_at TIMESTAMPTZ,

  created_at TIMESTAMPTZ DEFAULT now()
);
```

**Estimated Effort:** 4-5 days

---

### 2.2 Experiment Tracking

**Why:** Need to log which parameters were used for each analysis so experiments are reproducible.

**What to Build:**

- Store preprocessing parameters with each analysis:
  - Bandpass range, notch frequency
  - Segmentation window size, overlap
  - Feature dimension, embedding dimension
  - Model version used

- Query interface:
  - Filter analyses by preprocessing parameters
  - Compare results across parameter settings
  - Export experiment data for publication

**Expected Impact:**

- ✅ Reproducible research (can rerun experiment with same params)
- ✅ Ablation studies (test effect of each parameter)
- ✅ Scientific transparency
- ✅ Data for meta-analysis

**SQL Extension:**

```sql
ALTER TABLE eeg_analyses ADD COLUMN experiment_id UUID;
ALTER TABLE eeg_analyses ADD COLUMN preprocessing_params JSONB;

CREATE TABLE experiments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id),
  name TEXT NOT NULL,
  description TEXT,
  preprocessing_params JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

**Estimated Effort:** 3-4 days

---

### 2.3 Artifact Rejection

**Why:** Eye blinks, muscle noise contaminate embeddings. Need automatic detection.

**What to Build:**

- Implement amplitude thresholding:
  - Flag samples > 5 standard deviations
  - Log flagged window count
  - Option to reject analysis if >10% contaminated

- Add to preprocessing pipeline:
  ```typescript
  export function detectArtifacts(data: number[][], threshold = 5): number[] {
    // Returns boolean array of flagged samples
  }
  ```

**Expected Impact:**

- ✅ Better embedding quality
- ✅ Downstream ML models will be more accurate
- ✅ Reproducible artifact handling

**Estimated Effort:** 2-3 days

---

### 2.4 Signal Quality Metrics

**Why:** Need to assess whether input signal is suitable for analysis.

**What to Build:**

- Per-analysis quality score combining:
  - Signal-to-noise ratio (SNR)
  - Flatness detection (electrode disconnection)
  - Clipping detection (saturation)
  - Artifact percentage (from detector above)

- Return quality warnings with results:
  ```json
  {
    "signal_quality": {
      "snr_db": 12.5,
      "flatness_channels": ["Cz", "Pz"],
      "clipping_percent": 2.1,
      "artifact_percent": 5.3,
      "overall_score": 0.78,
      "warnings": ["Low SNR on Pz", "5% clipping detected"]
    }
  }
  ```

**Expected Impact:**

- ✅ Users know if their data is bad
- ✅ Can reject analyses that won't be reliable
- ✅ Transparency about data quality

**Estimated Effort:** 3-4 days

---

## PHASE 3: SCIENTIFIC VALIDATION (Months 4-8)

### 3.1 Ground Truth Annotation

**Why:** Cognitive metrics are unvalidated. Need to compare against physiological ground truth.

**What to Build:**

- Web interface for annotating analyses:
  - Video playback of task during EEG recording
  - Experimenter can mark:
    - Attention episodes (looking at target)
    - Workload changes (task difficulty)
    - Arousal changes (engagement level)

- Database schema:
  ```sql
  CREATE TABLE ground_truth_labels (
    id UUID PRIMARY KEY,
    analysis_id UUID REFERENCES eeg_analyses(id),
    label_type TEXT,  -- 'attention', 'workload', 'arousal'
    value FLOAT8,  -- 0-1
    start_sample INT,
    end_sample INT,
    annotator_user_id UUID REFERENCES auth.users(id),
    confidence FLOAT8
  );
  ```

**Expected Impact:**

- ✅ Can validate heuristic metrics
- ✅ Can train ML models with labeled data
- ✅ Foundation for publishing results

**Estimated Effort:** 4-6 weeks (UI + analysis design)

---

### 3.2 Cross-Subject Validation

**Why:** Need to test generalization across subjects.

**What to Build:**

- Leave-one-subject-out (LOSO) evaluation framework:

  ```typescript
  async function evaluateLOSO(
    analyses: Analysis[],
    subjects: string[],
  ): Promise<{
    auc: number;
    f1: number;
    per_subject: Record<string, { auc: number; f1: number }>;
  }> {
    // For each subject:
    //   1. Train on other subjects
    //   2. Test on held-out subject
    //   3. Collect metrics
  }
  ```

- API endpoint:
  - `POST /api/evaluate/cross-subject` — run LOSO evaluation
  - Returns per-subject metrics + aggregate

**Expected Impact:**

- ✅ Know if metrics generalize across people
- ✅ Identify individual differences
- ✅ Honest assessment of system performance

**Estimated Effort:** 2-3 days (assuming ground truth exists)

---

### 3.3 Benchmark Comparison

**Why:** Need to compare against published baselines.

**What to Build:**

- Test on standard datasets:
  - PhysioNet (already supported)
  - BCI Competition IV (dataset loader ready)
  - Sleep-EDF (implement loader)
  - CHB-MIT (implement loader)

- Compute metrics:
  - Classification accuracy (if labels provided)
  - Feature separability (Fisher's linear discriminant)
  - Correlation with published baselines

**Expected Impact:**

- ✅ Can cite performance vs. published work
- ✅ Identifies dataset-specific performance
- ✅ Foundation for paper publication

**Estimated Effort:** 4-6 weeks

---

### 3.4 Statistical Reporting

**Why:** Need proper statistical rigor for scientific claims.

**What to Build:**

- Compute for each metric:
  - Mean ± std across subjects
  - 95% confidence intervals
  - p-values (against null hypothesis of chance performance)
  - Effect sizes (Cohen's d)
  - Correlation with alternative measures

- Generate publication-ready tables:
  ```markdown
  | Metric    | Mean | Std  | 95% CI      | p-value |
  | --------- | ---- | ---- | ----------- | ------- |
  | Attention | 0.72 | 0.15 | (0.68-0.76) | <0.001  |
  ```

**Expected Impact:**

- ✅ Can submit to peer-reviewed journals
- ✅ Results are reproducible and verifiable
- ✅ Scientific credibility

**Estimated Effort:** 1-2 weeks

---

## PHASE 4: MACHINE LEARNING (Months 8-12)

### 4.1 ML Infrastructure Setup

**Why:** Currently no training infrastructure. To make real AI claims, need to build it.

**What to Build:**

- TensorFlow.js or ONNX Runtime for inference
- Training pipeline with:
  - Dataset loading from Supabase
  - Train/val/test splitting
  - Loss functions (cross-entropy for classification, MSE for regression)
  - Optimizers (Adam, SGD)
  - Checkpointing
  - Early stopping

**Example:**

```typescript
import * as tf from "@tensorflow/tfjs";

async function trainEmbeddingModel(trainingData) {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ units: 128, activation: "relu", inputShape: [C * 5] }),
      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({ units: 64, activation: "relu" }),
      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({ units: latentDim }),
    ],
  });

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "meanSquaredError",
  });

  await model.fit(trainingData.x, trainingData.y, {
    epochs: 100,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: [new tf.callbacks.EarlyStopping({ monitor: "val_loss", patience: 10 })],
  });

  // Save model
  await model.save("indexeddb://embedding-v1.0");
}
```

**Expected Impact:**

- ✅ Can now make legitimate AI claims
- ✅ Better embeddings than PCA
- ✅ Foundation for deep learning

**Estimated Effort:** 2-3 weeks

---

### 4.2 Deep Embedding Model

**Why:** PCA is shallow; deep autoencoders can learn richer representations.

**What to Build:**

- Variational autoencoder (VAE) for embeddings:

  ```
  Input → Dense(128) → ReLU → Dense(64) →
  Split → Mean & LogVar (latent) →
  Sample → Dense(64) → ReLU → Dense(128) → Output
  ```

- Loss = reconstruction + KL divergence

- Train on unlabeled EEG data (self-supervised)

**Expected Impact:**

- ✅ Better feature representations
- ✅ Can cluster similar EEG patterns
- ✅ Interpretable latent space

**Estimated Effort:** 4-6 weeks

---

### 4.3 Cognitive State Classifier

**Why:** Replace heuristics with trained classifier.

**What to Build:**

- Neural network classifier:

  ```
  Embedding (latent) → Dense(64) → ReLU → Dense(32) →
  Dense(3) → Softmax → [Attention, Workload, Arousal]
  ```

- Train on ground truth annotations (from Phase 3.1)
- Use cross-entropy loss

**Expected Impact:**

- ✅ Learned model that captures cognitive states
- ✅ Can validate against ground truth
- ✅ Publishable results

**Estimated Effort:** 3-4 weeks

---

### 4.4 Hyperparameter Optimization

**Why:** Manual tuning won't find optimal settings.

**What to Build:**

- Bayesian optimization over:
  - Embedding dimensions (32-256)
  - Classifier hidden layers (1-3)
  - Dropout rates (0-0.5)
  - Learning rates (1e-5 to 1e-2)
  - Regularization strength

- Use Optuna or Hyperband framework

**Expected Impact:**

- ✅ Best-possible model performance
- ✅ Reproducible hyperparameter selection
- ✅ Published results that are properly tuned

**Estimated Effort:** 2-3 weeks

---

## IMPLEMENTATION ROADMAP

```
Month 1-2: Foundation
├─ Database schema (1 week)
├─ Auth integration (1 week)
├─ Security hardening (1 week)
└─ Testing & deployment (1 week)

Month 2-4: Scalability
├─ Model versioning (1 week)
├─ Experiment tracking (1 week)
├─ Artifact rejection (1 week)
├─ Signal quality metrics (1 week)
└─ Testing & deployment (1 week)

Month 4-8: Scientific Validation
├─ Ground truth annotation UI (3 weeks)
├─ LOSO evaluation (1 week)
├─ Benchmark comparison (3 weeks)
├─ Statistical analysis (1 week)
└─ Publication prep (2 weeks)

Month 8-12: Machine Learning
├─ ML infrastructure (2 weeks)
├─ Deep embedding model (4 weeks)
├─ Cognitive classifier (3 weeks)
├─ Hyperparameter tuning (2 weeks)
└─ Validation & deployment (2 weeks)
```

---

## RESOURCE REQUIREMENTS

### Team

- **Backend Engineer:** 1 FTE (Supabase, Node.js)
- **ML Engineer:** 1 FTE (TensorFlow/PyTorch)
- **Neuroscientist/Validation:** 0.5 FTE (experiment design, ground truth)
- **DevOps:** 0.5 FTE (deployment, monitoring)

### Infrastructure

- Supabase (PostgreSQL + auth): $200-500/month
- GPU for training (optional): $100-1000/month
- Cloud storage: $50-200/month

### External

- fMRI data for ground truth: negotiate with collaborators
- Video recording setup: $500-2000 (one-time)
- Subject recruitment: $5000-10000 per study

---

## SUCCESS METRICS

### Phase 1 (Foundation)

- ✅ Users can retrieve past analyses
- ✅ 100% API endpoint coverage with auth
- ✅ Zero security issues in penetration test

### Phase 2 (Scalability)

- ✅ Model versions tracked in registry
- ✅ Experiment parameters logged for all analyses
- ✅ Artifact rejection reduces embedding variance by 20%

### Phase 3 (Validation)

- ✅ >50 subjects with ground truth labels
- ✅ LOSO validation shows >0.8 AUC
- ✅ Results publishable in peer-reviewed journal

### Phase 4 (ML)

- ✅ Deep model achieves >90% accuracy on benchmark datasets
- ✅ Hyperparameter optimization improves test AUC by 10%
- ✅ Can make legitimate AI claims

---

## CONCLUSION

Current codebase provides solid foundation for EEG signal processing. Roadmap outlines path to production neurotechnology platform with validated AI components.

**Key Dependencies:**

1. Authentication (blocks Phases 1-4)
2. Ground truth (blocks Phases 3-4)
3. ML infrastructure (blocks Phase 4)

**Critical Decision Point (Month 4):**

- If Phase 3 validation fails → recalibrate heuristics, focus on Phase 2 scalability
- If validation succeeds → proceed to Phase 4 ML with confidence
