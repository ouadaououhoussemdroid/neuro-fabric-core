# EEGConformer — Artifact Acquisition Report

- Date: 2026-06-17
- Scope: locate a production-usable pretrained **EEGConformer** checkpoint
  for Neuro-Fabric's first real EEG foundation-model deployment.
- Status: **No drop-in pretrained checkpoint exists.** Acquisition path
  requires either (a) training in-house on an open EEG corpus, or
  (b) reproducing a published authors' checkpoint under its dataset
  licence.
- Non-goals: EEGPT, EEG2IMG, architectural redesign.

## 1. Ecosystem search

| Source | Finding | Evidence |
|---|---|---|
| **Braindecode** (`braindecode>=0.8`) | Ships the `EEGConformer` **architecture**, no pretrained weights. `braindecode.models` exposes constructors only; no `from_pretrained()`, no model-zoo bucket. | `braindecode.models.EEGConformer` docstring; absence of weights in `braindecode/models/__init__.py` |
| **MOABB** | Distributes **datasets** (BCI-IV-2a/2b, BNCI, PhysioNet MI, Schirrmeister 2017, etc.), not weights. | moabb.datasets |
| **MNE / mne-tools** | No EEGConformer weights. | n/a |
| **HuggingFace Hub** | As of 2026-06, no widely-adopted `EEGConformer` checkpoint with a permissive licence and reproducible training card. A handful of personal forks exist; none are citeable or licence-clean. | manual review |
| **Paper authors' repo** (Song, Zheng, Ko 2022) | Reference PyTorch code published; **no released weights**. Reproduction recipes target BCI-IV-2a / SEED / SEED-IV. | github.com/eeyhsong/EEG-Conformer |
| **Papers-with-Code** | Architecture entry exists; no associated checkpoint download. | paperswithcode.com |
| **Zenodo / OSF** | No canonical EEGConformer release. Some lab-specific fine-tunes exist but are paradigm-locked (motor-imagery only). | search 2026-06 |

**Conclusion:** there is no equivalent of "BERT-base on HuggingFace" for
EEGConformer in 2026-06. This matches the broader EEG-foundation-model
landscape where EEGPT, LaBraM, and BIOT dominate the *pretrained* niche
and EEGConformer remains a *trainable architecture*.

## 2. Legal / licence assessment

| Path | Code licence | Data licence | Net redistribution |
|---|---|---|---|
| Train on **BCI-IV-2a** (Graz) | Braindecode = BSD-3 | BCI-IV "free for academic and commercial use with citation" | ✅ Weights redistributable with attribution |
| Train on **PhysioNet MI** (Schalk 2004) | BSD-3 | ODC-BY 1.0 | ✅ Redistributable with attribution |
| Train on **SEED / SEED-IV** | BSD-3 | Research-only, signed EULA | ❌ Cannot ship weights publicly |
| Train on **TUH EEG** | BSD-3 | Research-only DUA | ❌ Cannot ship weights publicly |
| Reproduce author recipe on BCI-IV-2a | BSD-3 | as above | ✅ Same as row 1 |

**Recommended licence posture for the first shipped artefact:**
train on **BCI-IV-2a** (and optionally PhysioNet MI for cross-paradigm
generality). Ship the resulting `eegconformer.onnx` under **CC-BY-4.0**
with a `MODEL_CARD.md` citing Braindecode, the dataset, and the training
recipe.

## 3. Compatibility with current contract

The production contract from
`docs/audits/2026-06-17_braindecode-model-selection.md` is:
22 channels · 250 Hz · 1000 samples · 32-D attention-pooled embedding ·
opset 17.

| Dataset | Channels | Native Hz | Native window | Adaptation required |
|---|---:|---:|---:|---|
| BCI-IV-2a | 22 | 250 | 4 s | **None** — matches contract verbatim |
| PhysioNet MI | 64 | 160 | variable | Resample to 250 Hz, select 22-channel subset |
| SEED | 62 | 200 | 1 s | Channel remap + resample (and licence blocks redistribution) |

BCI-IV-2a is the obvious target: zero contract drift, permissive licence,
small (≈ 600 MB raw), and is already the canonical Braindecode tutorial
dataset.

## 4. Acquisition plan

Two parallel tracks, A first.

### Track A — Reference checkpoint (now)

1. Download BCI-IV-2a via MOABB (`MOABB → moabb.datasets.BNCI2014001`).
2. Run the canonical Braindecode EEGConformer tutorial unchanged
   (cross-session, 4-class, 4 s @ 250 Hz).
3. Export via `scripts/export_braindecode_eegconformer.py` (already in
   the repo). Confirms PyTorch↔ONNX parity ≥ 0.999.
4. Ship as `eegconformer-bciiv2a-v1.onnx` under CC-BY-4.0.

Expected effort: **~2 engineer-days** (1 day training, 1 day validation
+ benchmarking).

### Track B — Cross-paradigm checkpoint (next)

1. Pre-train on PhysioNet MI (after channel remap to the 22-ch BCI-IV-2a
   montage and 160→250 Hz resample), then fine-tune on BCI-IV-2a.
2. Expected to improve cosine recall@10 by 5–10 pp on the holdout, at
   the cost of ~1 extra engineer-day.

Both tracks reuse the *same* artefact-acquisition pipeline; B is purely
a data-mix change.

## 5. Risk register (acquisition-only)

| Risk | Likelihood | Impact | Mitigation |
|---|:---:|:---:|---|
| Training instability on small dataset | Medium | Medium | Use Braindecode reference recipe verbatim; fixed seeds |
| Overfitting to BCI-IV-2a | High | Medium | Document scope; gate on cross-subject holdout; plan Track B |
| ONNX export drift (opset / attention) | Low | Medium | Pin `torch==2.3`, `opset=17`, smoke test in script |
| Licence misclassification | Low | High | Pre-flight check in MODEL_CARD before publish |
| Embedding head changes between runs | Medium | Low | Pin hook target to `model.fc`; assert tensor shape in exporter |

## 6. Expected embedding quality (a-priori)

- Within-paradigm cosine recall@10 (BCI-IV-2a holdout): **0.55–0.70**
  vs PCA baseline ~0.30–0.40 (qualitative; to be confirmed by
  `benchmarkAll`).
- Cross-paradigm (Track A → unseen dataset): likely degrades to
  ~0.35–0.45 — close to PCA. Track B addresses this.
- Latency / memory profile: already measured in
  `docs/audits/2026-06-17_braindecode-benchmark.md` (P50 ~ 305 ms,
  P95 ~ 400 ms, heap Δ ~ 19 MB on WASM SIMD).

## 7. Single highest-impact next action

Run Track A end-to-end and publish `eegconformer-bciiv2a-v1.onnx`. This
converts the *entire* AI Foundation Layer from "ready" to "live" without
touching a line of TypeScript.