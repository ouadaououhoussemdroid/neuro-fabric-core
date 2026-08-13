#!/usr/bin/env python3
"""
T-034: Representation Learning Experiment — Training Script

Tests whether better training objectives (EEG augmentation + supervised contrastive
loss) can improve the 32-D EEGConformer embedding without changing the architecture.

Experiments:
    baseline          — CE + label_smoothing (reproduces v2)
    aug               — CE + label_smoothing + EEG augmentation
    contrastive       — CE + label_smoothing + SupCon loss (λ=0.5)
    aug_contrastive   — CE + label_smoothing + augmentation + SupCon (λ=0.5)

All use identical hyperparameters to v2 (LR 5e-5, WD 1e-3, batch 32, epochs 200,
warmup 15, label_smoothing 0.1, seed 20260617). Only the loss function and
augmentation change.

Production safety:
    - Does NOT modify production models (eegconformer.onnx, eegconformer_finetuned.onnx)
    - Does NOT change the 32-D API contract
    - All models saved to training/artefacts/eegconformer-t034-{config}/
    - All models exported to ONNX with parity verification (>0.999 cosine)

Usage:
    python scripts/t034-train-representation.py --experiment baseline --out-dir training/artefacts/eegconformer-t034-baseline
    python scripts/t034-train-representation.py --experiment contrastive --contrastive-weight 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ─── Constants (identical to v2) ──────────────────────────────────────────────────

EEGCONFORMER_CHANS = [
    "FP1", "FP2", "F5", "F6", "F3", "F4", "F1", "F2",
    "FC5", "FC6", "FC3", "FC4", "C5", "C6", "C3", "C4",
    "T7", "T8", "P7", "P8", "P5", "P6",
]

SAMPLE_RATE = 250
WINDOW_SAMPLES = 1000
BANDPASS = [4.0, 38.0]
N_SECONDS = 4.0
CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]

BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]


# ─── EEGConformer with embedding extraction ─────────────────────────────────────────

class EEGConformerWithEmbedding(nn.Module):
    """Wrapper that exposes both the 32-D embedding and 4-D logits.

    Mirrors braindecode's EEGConformer.forward inline so the training loop
    can access the 32-D `fc` output for contrastive loss computation.

    Forward path:
        unsqueeze(1) → patch_embedding → transformer → fc (32-D) → final_layer (4-D)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # x: [B, 22, 1000]
        x = torch.unsqueeze(x, dim=1)       # [B, 1, 22, 1000]
        x = self.model.patch_embedding(x)    # [B, 61, 40]
        feature = self.model.transformer(x)   # [B, 61, 40]
        embedding = self.model.fc(feature)   # [B, 32]
        logits = self.model.final_layer(embedding)  # [B, 4]
        return embedding, logits


# ─── Supervised Contrastive Loss ────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Pulls same-class embeddings together and pushes different-class apart.
    Operates on L2-normalized features with a temperature hyperparameter.

    Formula:
        L = -1/|P_i| * sum_{j in P_i} log( exp(sim(z_i, z_j)/τ) / sum_{k≠i} exp(sim(z_i, z_k)/τ) )

    Where P_i = same-class samples excluding i itself.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Args:
            features: [B, D] — L2-normalized embedding (32-D)
            labels:   [B] — class labels
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # L2 normalize features
        features = F.normalize(features, p=2, dim=1)

        # Cosine similarity matrix: [B, B]
        sim = torch.matmul(features, features.T) / self.temperature

        # For numerical stability, subtract max (log-sum-exp trick)
        logits = sim - torch.max(sim.detach(), dim=1, keepdim=True)[0]

        # Exponential similarities
        exp_sim = torch.exp(logits)

        # Zero out diagonal (self-similarity)
        mask_diag = torch.ones_like(exp_sim) - torch.eye(batch_size, device=device)
        exp_sim = exp_sim * mask_diag

        # Sum over all (excluding self): denominator
        sum_exp = exp_sim.sum(dim=1, keepdim=True)  # [B, 1]

        # Positive mask: same class, excluding self
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float() * mask_diag  # [B, B]

        # Log probability for positive pairs
        # log( exp(sim_ij) / sum_k exp(sim_ik) ) for each positive pair j
        log_prob = logits - torch.log(sum_exp + self.eps)

        # Mean over positive pairs for each anchor
        n_pos = mask_pos.sum(dim=1, keepdim=True)  # [B, 1]
        
        # Handle anchors with no positive pairs (same class only themselves in batch)
        has_pos = (n_pos > 0).float().squeeze()  # [B]
        
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (n_pos.squeeze() + self.eps)  # [B]
        
        # Loss
        loss = -mean_log_prob_pos * has_pos  # zero out samples with no positives
        loss = loss.sum() / (has_pos.sum() + self.eps)

        return loss


# ─── EEG Augmentations ──────────────────────────────────────────────────────────────

class EEGAugmentation(nn.Module):
    """Realistic EEG augmentations applied in the input space [B, C, T].

    All augmentations use conservative parameters appropriate for EEG:
    - Channel dropout: randomly zero out channels with probability p
    - Amplitude scaling: multiply by U[0.9, 1.1]
    - Gaussian noise: add noise with specified SNR
    - Time masking: zero out random time segments

    Reference: The braindecode EEGConformer docstring recommends "segmentation and
    recombination" augmentation before use. We use conservative parameters to avoid
    destroying signal.
    """

    def __init__(
        self,
        channel_dropout_p: float = 0.1,
        amplitude_scale_range: tuple = (0.9, 1.1),
        noise_snr_db: float = 20.0,
        time_mask_width: int = 50,        # 200ms at 250 Hz
        num_time_masks: int = 2,
        p: float = 0.8,                   # probability of applying augmentation per sample
    ):
        super().__init__()
        self.channel_dropout_p = channel_dropout_p
        self.amp_low, self.amp_high = amplitude_scale_range
        self.noise_snr_db = noise_snr_db
        self.time_mask_width = time_mask_width
        self.num_time_masks = num_time_masks
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, 22, 1000] or [B, 1, 22, 1000]
        """
        # Handle both input formats
        squeezed = False
        if x.dim() == 4:
            x = x.squeeze(1)  # [B, 22, 1000]
            squeezed = True

        B, C, T = x.shape
        device = x.device

        for i in range(B):
            if torch.rand(1, device=device).item() > self.p:
                continue

            # 1. Channel dropout
            if self.channel_dropout_p > 0:
                ch_mask = (torch.rand(C, device=device) > self.channel_dropout_p).float()
                # Ensure at least 1 channel remains
                if ch_mask.sum() == 0:
                    ch_mask[torch.randint(0, C, (1,), device=device)] = 1.0
                x[i] = x[i] * ch_mask.unsqueeze(1)

            # 2. Amplitude scaling
            if self.amp_low != self.amp_high:
                scale = torch.empty(1, device=device).uniform_(self.amp_low, self.amp_high)
                x[i] = x[i] * scale

            # 3. Gaussian noise (SNR-based)
            if self.noise_snr_db < 100:
                signal_power = x[i].pow(2).mean()
                noise_power = signal_power / (10 ** (self.noise_snr_db / 10))
                noise = torch.randn_like(x[i]) * torch.sqrt(noise_power + 1e-9)
                x[i] = x[i] + noise

            # 4. Time masking
            for _ in range(self.num_time_masks):
                if self.time_mask_width > 0:
                    t_start = torch.randint(0, T, (1,), device=device).item()
                    t_start = max(0, t_start - self.time_mask_width // 2)
                    t_end = min(T, t_start + self.time_mask_width)
                    x[i, :, t_start:t_end] = 0.0

        if squeezed:
            x = x.unsqueeze(1)  # restore [B, 1, 22, 1000]

        return x


# ─── Collapse Monitoring ───────────────────────────────────────────────────────────

def compute_embedding_stats(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Compute representation quality metrics on embeddings.

    Args:
        embeddings: [N, 32] L2-normalized or raw embeddings
        labels: [N] class labels
    """
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    n = embeddings.shape[0]

    stats = {}

    # L2 normalize for cosine-based metrics
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    emb_norm = embeddings / norms

    # 1. Per-dimension statistics
    stats["dim_mean"] = float(np.mean(embeddings, axis=0).mean())
    stats["dim_std"] = float(np.std(embeddings, axis=0).mean())
    stats["embedding_norm_mean"] = float(np.mean(norms))
    stats["embedding_norm_std"] = float(np.std(norms))

    # 2. Effective rank (participation ratio)
    # PR = (sum(var_i))^2 / sum(var_i^2)
    var = np.var(emb_norm, axis=0)
    pr = (var.sum() ** 2) / (var.pow(2).sum() if hasattr(var, 'pow') else (var ** 2).sum() + 1e-9)
    stats["effective_rank"] = float(pr)

    # 3. Intra/inter-class cosine similarity
    sims = emb_norm @ emb_norm.T  # [N, N]

    # Build label mask
    labels_2d = labels.reshape(-1, 1)
    same_class = (labels_2d == labels_2d.T)
    diff_class = ~same_class
    np.fill_diagonal(same_class, False)  # exclude self

    n_intra = same_class.sum()
    n_inter = diff_class.sum()

    intra_cos = sims[same_class].mean() if n_intra > 0 else 0.0
    inter_cos = sims[diff_class].mean() if n_inter > 0 else 0.0

    stats["intra_class_cosine"] = float(intra_cos)
    stats["inter_class_cosine"] = float(inter_cos)
    stats["separation_margin"] = float(intra_cos - inter_cos)  # lower is better (intra < inter)

    # 4. Fisher score (proxy: (inter - intra) / (inter_std + intra_std + eps))
    if n_intra > 0 and n_inter > 0:
        intra_vals = sims[same_class]
        inter_vals = sims[diff_class]
        fisher = (inter_vals.mean() - intra_vals.mean()) / (inter_vals.std() + intra_vals.std() + 1e-9)
        stats["fisher_score"] = float(abs(fisher))

    return stats


# ─── Data loading (copy from v2 for consistency) ────────────────────────────────────

def normalize_ch_name(ch: str):
    return ch.replace(".", "").upper()


def load_subject_trials(subj_code: str, data_dir: str):
    """Load EDF for a single subject, return (trials, labels, ch_names)."""
    import mne

    trials, labels, ch_names = [], [], []
    for run_idx, run in enumerate([5, 6]):
        fname = os.path.join(data_dir, subj_code, f"{subj_code}R{run:02d}.edf")
        if not os.path.exists(fname):
            continue
        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        if ch_names == []:
            ch_names = [normalize_ch_name(c) for c in raw.ch_names]
        sfreq = raw.info["sfreq"]
        events, _ = mne.events_from_annotations(raw, verbose=False)

        for ev in events:
            idx = np.argmin(np.abs(raw.annotations.onset - ev[0] / sfreq))
            event_type = raw.annotations.description[idx]
            if event_type not in ("T1", "T2"):
                continue
            onset = ev[0]
            trial_len = int(4.0 * sfreq)
            start = int(onset)
            end = min(start + trial_len, len(raw.times))
            trial = raw.get_data()[:, start:end]

            # Corrected label mapping (fixes T-031 ternary bug)
            if run_idx == 0:
                label = 0 if event_type == "T1" else 1
            else:
                label = 2 if event_type == "T1" else 3

            trials.append(trial.astype(np.float32))
            labels.append(label)

    return trials, labels, ch_names


def preprocess_trial(trial_data: np.ndarray, source_ch_names: list) -> np.ndarray:
    """Resample 160→250 Hz, bandpass 4-38 Hz, select 22 channels, z-score.

    Input:  [n_ch, n_samples] at 160 Hz
    Output: [22, 1000] float32
    """
    import mne

    source_idx = {normalize_ch_name(c): i for i, c in enumerate(source_ch_names)}
    selected = np.array([trial_data[source_idx[ch]] for ch in EEGCONFORMER_CHANS])

    info = mne.create_info(ch_names=EEGCONFORMER_CHANS, sfreq=160, ch_types="eeg")
    inst = mne.io.RawArray(selected, info, verbose=False)
    inst.resample(SAMPLE_RATE, verbose=False)
    inst.filter(BANDPASS[0], BANDPASS[1], verbose=False, method="fir", fir_design="firwin")
    data = inst.get_data()

    target = WINDOW_SAMPLES
    if data.shape[1] < target:
        pad = target - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad)), mode="edge")
    elif data.shape[1] > target:
        start = (data.shape[1] - target) // 2
        data = data[:, start:start + target]

    # Z-score per channel
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-6
    return ((data - mean) / std).astype(np.float32)


def load_onnx_weights(pt_model: nn.Module, onnx_path: str) -> dict:
    """Extract weights from the existing BCI-IV-2a ONNX model.

    Identical to v2's load_onnx_weights — reconstructs PyTorch state_dict from ONNX.
    """
    import onnx
    import onnx.numpy_helper as nh

    onnx_model = onnx.load(onnx_path)
    onnx_init = {}
    for init in onnx_model.graph.initializer:
        onnx_init[init.name] = nh.to_array(init)

    pt_state = pt_model.state_dict()
    state_load = {}

    # 1. Directly matched weights (strip 'model.' prefix)
    for k, v in pt_state.items():
        onnx_key = f"model.{k}"
        if onnx_key in onnx_init and onnx_init[onnx_key].shape == tuple(v.shape):
            state_load[k] = torch.from_numpy(onnx_init[onnx_key].copy())

    # 2. BatchNorm2d → identity (shallownet.2)
    for k, v in pt_state.items():
        if "shallownet.2" in k and k not in state_load:
            if "weight" in k:
                state_load[k] = torch.ones_like(v)
            elif "bias" in k:
                state_load[k] = torch.zeros_like(v)
            elif "running_mean" in k or "running_var" in k:
                state_load[k] = torch.zeros_like(v)
            elif "num_batches_tracked" in k:
                state_load[k] = torch.zeros_like(v)

    # 3. QKV weights from _v_ entries (40, 120)
    v_40_120 = sorted(
        [(k, v) for k, v in onnx_init.items() if k.startswith("val_") and v.shape == (40, 120)],
        key=lambda x: int(x[0].split("_")[1]),
    )
    pt_state_keys = list(pt_state.keys())
    qkv_idx = 0
    for k, v in pt_state.items():
        if "in_proj_weight" in k and k not in state_load:
            if qkv_idx < len(v_40_120):
                varr = v_40_120[qkv_idx][1]
                state_load[k] = torch.from_numpy(varr.T.copy())
                qkv_idx += 1

    # 4. Projection weights from val_ (40, 40) entries
    v_40_40 = sorted(
        [(k, v) for k, v in onnx_init.items() if k.startswith("val_") and v.shape == (40, 40)],
        key=lambda x: int(x[0].split("_")[1]),
    )
    proj_idx = 0
    for k, v in pt_state.items():
        if "proj.weight" in k and k not in state_load:
            if proj_idx < len(v_40_40):
                varr = v_40_40[proj_idx][1]
                # Reshape from (1, 1, 40, 40) to (40, 40)
                state_load[k] = torch.from_numpy(varr.reshape(40, 40).T.copy())
                proj_idx += 1

    # 5. FFN first layer from val_ (40, 160) entries
    val_40_160 = sorted(
        [(k, v) for k, v in onnx_init.items() if k.startswith("val_") and v.shape == (40, 160)],
        key=lambda x: int(x[0].split("_")[1]),
    )
    for idx, (vname, varr) in enumerate(val_40_160):
        pt_key = f"transformer.{idx}.1.fn.1.0.weight"
        if pt_key not in state_load:
            state_load[pt_key] = torch.from_numpy(varr.T.copy())

    # 6. FFN second layer from val_ (160, 40) entries
    val_160_40 = sorted(
        [(k, v) for k, v in onnx_init.items() if k.startswith("val_") and v.shape == (160, 40)],
        key=lambda x: int(x[0].split("_")[1]),
    )
    for idx, (vname, varr) in enumerate(val_160_40):
        pt_key = f"transformer.{idx}.1.fn.1.3.weight"
        if pt_key not in state_load:
            state_load[pt_key] = torch.from_numpy(varr.T.copy())

    return state_load


def set_deterministic(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_warmup_cosine_lr(step, warmup_steps, total_steps, base_lr):
    """Linear warmup + cosine annealing."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ─── ONNX Export ────────────────────────────────────────────────────────────────────

def export_to_onnx(model, ckpt_path: str, output_path: str, device: torch.device):
    """Export the model to ONNX with embedding + logits outputs.

    Uses the same wrapper pattern from scripts/export_braindecode_eegconformer.py
    (EEGConformerExportWrapper). The forward path is identical:
    unsqueeze → patch_embedding → transformer → fc → final_layer.
    """
    # Load best checkpoint
    state = torch.load(ckpt_path, map_location="cpu")
    model.model.load_state_dict(state)
    wrapper = EEGConformerWithEmbedding(model.model)
    wrapper.eval()
    wrapper.to("cpu")

    dummy = torch.randn(1, 22, 1000)

    # PyTorch reference
    with torch.no_grad():
        pt_emb, pt_logits = wrapper(dummy)

    # Export
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["embedding", "logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "embedding": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Parity check
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_emb, ort_logits = sess.run(None, {"input": dummy.numpy()})

    cos_emb = torch.nn.functional.cosine_similarity(
        pt_emb.flatten().unsqueeze(0),
        torch.from_numpy(ort_emb).flatten().unsqueeze(0),
    ).item()
    cos_logits = torch.nn.functional.cosine_similarity(
        pt_logits.flatten().unsqueeze(0),
        torch.from_numpy(ort_logits).flatten().unsqueeze(0),
    ).item()

    print(f"  ONNX export: {output_path}")
    print(f"  Parity: embedding cosine={cos_emb:.6f}, logits cosine={cos_logits:.6f}")
    assert cos_emb > 0.999, f"Embedding parity failed: {cos_emb}"
    assert cos_logits > 0.999, f"Logits parity failed: {cos_logits}"

    # WASM compatibility check
    import onnx
    onnx_model = onnx.load(output_path)
    ops_used = set()
    for node in onnx_model.graph.node:
        ops_used.add(node.op_type)
    wasm_blockers = {"DFT", "ReduceL2", "FFT", "Complex", "GlobalAveragePool", "Flatten"}
    blocked = wasm_blockers & ops_used
    if blocked:
        print(f"  WARNING: WASM blockers: {blocked}")
    else:
        print(f"  WASM compatible: all ops OK")


# ─── Main Training ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--train-subj", type=int, nargs="+", default=list(range(6, 41)))
    ap.add_argument("--test-subj", type=int, nargs="+", default=list(range(41, 51)))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--onnx-init", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--experiment", type=str, default="baseline",
                    choices=["baseline", "aug", "contrastive", "aug_contrastive"],
                    help="Training experiment type")
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="SupCon temperature")
    ap.add_argument("--contrastive-weight", type=float, default=0.5,
                    help="Weight for SupCon loss (lambda)")
    ap.add_argument("--no-onnx-init", action="store_true",
                    help="Skip ONNX weight initialization (random init)")
    ap.add_argument("--export-onnx", action="store_true", default=True,
                    help="Export to ONNX after training")
    ap.add_argument("--no-export-onnx", action="store_true", default=False,
                    help="Skip ONNX export after training")
    args = ap.parse_args()

    # Set seeds
    set_deterministic(args.seed)

    # Resolve paths
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir or os.path.join(os.environ.get("TMP", "/tmp"), "eegmmidb")
    if not os.path.exists(data_dir):
        print(f"[t034] ERROR: data dir not found: {data_dir}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else \
        repo_root / "training" / "artefacts" / f"eegconformer-t034-{args.experiment}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[t034] Device: {device}")
    print(f"[t034] Experiment: {args.experiment}")

    # ─── Load and preprocess data (with caching) ─────────────────────────────
    train_subj = sorted(args.train_subj)
    test_subj = sorted(args.test_subj)
    all_subj = sorted(set(train_subj) | set(test_subj))

    cache_path = os.path.join(os.path.dirname(data_dir), "eegmmidb_t034_cached.npz")
    print(f"[t034] Loading data: train={train_subj}, test={test_subj}")

    # Build a cache key from the subject list
    cache_key = f"subjects_{'_'.join(str(s) for s in all_subj)}"
    cache_path = os.path.join(os.path.dirname(data_dir), "eegmmidb_t034_cached.npz")
    loaded_from_cache = False

    if os.path.exists(cache_path):
        try:
            cache = np.load(cache_path, allow_pickle=True)
            cache_subjs = set(cache["s_all"].tolist())
            cache_source_ch = list(cache["source_ch_names"])
            if cache_key in cache and set(all_subj).issubset(cache_subjs):
                print(f"[t034] Loading from cache: {cache_path} (key={cache_key})")
                X_all = cache[f"X_{cache_key}"]
                y_all = cache[f"y_{cache_key}"]
                s_all = cache[f"s_{cache_key}"]
                source_ch_names = cache_source_ch
                loaded_from_cache = True
                print(f"  Cache loaded: {len(X_all)} trials, {len(set(s_all))} subjects")
        except Exception as e:
            print(f"  Cache load failed: {e}, loading from EDF")

    if not loaded_from_cache:
        subj_data = {}
        source_ch_names = None
        for subj_id in all_subj:
            subj_code = f"S{subj_id:03d}"
            trials, labels, ch_names = load_subject_trials(subj_code, data_dir)
            if source_ch_names is None and ch_names:
                source_ch_names = ch_names
            if len(trials) == 0:
                print(f"  {subj_code}: SKIP (no trials)")
                continue
            processed, valid_labels = [], []
            for trial, label in zip(trials, labels):
                try:
                    proc = preprocess_trial(trial, source_ch_names if source_ch_names else ch_names)
                    processed.append(proc)
                    valid_labels.append(label)
                except Exception as e:
                    print(f"  {subj_code}: WARN preprocessing error: {e}")
                    continue
            if len(processed) > 0:
                subj_data[subj_id] = {
                    "X": np.stack(processed),
                    "y": np.array(valid_labels, dtype=np.int64),
                }
                print(f"  {subj_code}: {len(valid_labels)} trials OK")

        # Combine all subjects
        Xs = [subj_data[s]["X"] for s in sorted(subj_data)]
        ys = [subj_data[s]["y"] for s in sorted(subj_data)]
        ss = [np.full(len(subj_data[s]["y"]), s, dtype=np.int64) for s in sorted(subj_data)]
        X_all = np.concatenate(Xs)
        y_all = np.concatenate(ys)
        s_all = np.concatenate(ss)

        # Save to cache for future runs
        try:
            np.savez(cache_path,
                     X_all=X_all, y_all=y_all, s_all=s_all,
                     source_ch_names=np.array(source_ch_names, dtype=object),
                     cache_key_str=cache_key,
                     **{f"X_{cache_key}": X_all, f"y_{cache_key}": y_all, f"s_{cache_key}": s_all})
            print(f"  Saved cache: {cache_path}")
        except Exception as e:
            print(f"  Cache save failed: {e}")

        print(f"  Total: {len(X_all)} trials, {len(set(s_all))} subjects")

    # ─── Split ─────────────────────────────────────────────────────────────
    available_subj = sorted(set(s_all.tolist()))
    # Internal 85/15 val split within train subjects
    train_ids = sorted([s for s in available_subj if s in train_subj])
    rng = np.random.RandomState(args.seed)
    # Shuffle for val split
    train_ids_shuffled = train_ids.copy()
    rng.shuffle(train_ids_shuffled)
    n_val = max(1, len(train_ids_shuffled) // 7)  # ~15%
    val_ids = sorted(train_ids_shuffled[:n_val])
    train_ids = sorted(train_ids_shuffled[n_val:])
    test_ids = sorted([s for s in available_subj if s in test_subj])

    print(f"\n[t034] Split: train={len(train_ids)} subjects, val={len(val_ids)} subjects, test={len(test_ids)} subjects")

    def build_split(ids):
        if not ids:
            return None
        mask = np.isin(s_all, ids)
        return X_all[mask], y_all[mask]

    X_train, y_train = build_split(train_ids)
    X_val, y_val = build_split(val_ids)
    X_test, y_test = build_split(test_ids)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ─── Build model ──────────────────────────────────────────────────────
    from braindecode.models import EEGConformer

    model = EEGConformer(
        n_outputs=4,
        n_chans=22,
        n_times=1000,
        final_fc_length="auto",
        drop_prob=args.dropout,
        att_drop_prob=args.dropout,
    )
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Initialise from ONNX (same as v2)
    onnx_path = args.onnx_init
    if onnx_path is None and not args.no_onnx_init:
        onnx_path = str(repo_root / "public" / "models" / "eegconformer.onnx")
    if onnx_path and os.path.exists(onnx_path):
        print(f"[t034] Initialising from ONNX: {onnx_path}")
        state_dict = load_onnx_weights(model, onnx_path)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"  Missing keys: {result.missing_keys}")

    model = EEGConformerWithEmbedding(model)
    model.to(device)

    # ─── Training setup ───────────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size, shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    supcon_loss_fn = SupConLoss(temperature=args.temperature)
    augment = EEGAugmentation()
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Calculate total optimizer steps
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup * steps_per_epoch

    # Training state
    best_val_loss = math.inf
    best_val_acc = 0.0
    best_epoch = 0
    best_test_acc = 0.0
    stale = 0
    ckpt_path = out_dir / "eegconformer.pt"
    history = []

    # Experiment flags
    use_aug = args.experiment in ("aug", "aug_contrastive")
    use_contrastive = args.experiment in ("contrastive", "aug_contrastive")
    contrastive_weight = args.contrastive_weight if use_contrastive else 0.0

    print(f"\n[t034] Training config:")
    print(f"  CE: yes, Label smoothing: {args.label_smoothing}")
    print(f"  Augmentation: {'yes' if use_aug else 'no'}")
    print(f"  SupCon: {'yes' if use_contrastive else 'no'}, λ={contrastive_weight}, τ={args.temperature}")
    print(f"  LR={args.lr}, WD={args.weight_decay}, batch={args.batch_size}")
    print(f"  steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    global_step = 0
    # Track representation stats for the best epoch
    best_emb_stats = None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_con_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Apply augmentation if enabled
            if use_aug:
                xb = augment(xb)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    embedding, logits = model(xb)
                    ce_loss = loss_fn(logits, yb)

                    if use_contrastive:
                        emb_norm = F.normalize(embedding, p=2, dim=1)
                        con_loss = supcon_loss_fn(emb_norm, yb)
                        loss = ce_loss + contrastive_weight * con_loss
                    else:
                        loss = ce_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                embedding, logits = model(xb)
                ce_loss = loss_fn(logits, yb)

                if use_contrastive:
                    emb_norm = F.normalize(embedding, p=2, dim=1)
                    con_loss = supcon_loss_fn(emb_norm, yb)
                    loss = ce_loss + contrastive_weight * con_loss
                else:
                    loss = ce_loss

                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # LR scheduling
            global_step += 1
            lr_now = get_warmup_cosine_lr(global_step, warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            epoch_loss += loss.item()
            epoch_ce_loss += ce_loss.item()
            if use_contrastive:
                epoch_con_loss += con_loss.item()
            n_batches += 1

        # Validation
        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        all_val_emb = []
        all_val_labels = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                embedding, logits = model(xb)
                vloss += loss_fn(logits, yb).item() * len(yb)
                vcorrect += (logits.argmax(-1) == yb).sum().item()
                vtotal += len(yb)
                all_val_emb.append(embedding.cpu().numpy())
                all_val_labels.extend(yb.cpu().numpy().tolist())

        vloss /= max(1, vtotal)
        vacc = vcorrect / max(1, vtotal)

        # Test (monitoring only)
        test_acc = 0.0
        if y_test is not None:
            model.eval()
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                batch_size=args.batch_size, shuffle=False,
            )
            tloss, tcorrect, ttotal = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    _, logits = model(xb)
                    tloss += loss_fn(logits, yb).item() * len(yb)
                    tcorrect += (logits.argmax(-1) == yb).sum().item()
                    ttotal += len(yb)
            test_acc = tcorrect / max(1, ttotal)

        # Collapse monitoring on val embeddings
        val_emb = np.vstack(all_val_emb)
        val_labels = np.array(all_val_labels)
        emb_stats = compute_embedding_stats(val_emb, val_labels)

        # Track stats for best epoch
        if vloss < best_val_loss - 1e-4:
            best_val_loss = vloss
            best_val_acc = vacc
            best_epoch = epoch
            best_test_acc = test_acc
            best_emb_stats = emb_stats
            stale = 0
            torch.save(model.model.state_dict(), ckpt_path)
        else:
            stale += 1

        epoch_loss_avg = epoch_loss / n_batches
        epoch_ce_avg = epoch_ce_loss / n_batches
        epoch_con_avg = epoch_con_loss / n_batches if use_contrastive else 0.0

        # Record history with collapse metrics
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss_avg,
            "train_ce_loss": epoch_ce_avg,
            "train_con_loss": epoch_con_avg,
            "val_loss": vloss,
            "val_acc": vacc,
            "test_acc": test_acc,
            "lr": lr_now,
            "emb_norm": emb_stats["embedding_norm_mean"],
            "effective_rank": emb_stats["effective_rank"],
            "intra_class_cosine": emb_stats["intra_class_cosine"],
            "inter_class_cosine": emb_stats["inter_class_cosine"],
            "separation_margin": emb_stats["separation_margin"],
            "fisher_score": emb_stats.get("fisher_score", 0.0),
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  ep={epoch:03d}: train_loss={epoch_loss_avg:.4f} (CE={epoch_ce_avg:.4f}, "
                  f"Con={epoch_con_avg:.4f}) val_loss={vloss:.4f} val_acc={vacc:.4f} "
                  f"test_acc={test_acc:.4f} eff_rank={emb_stats['effective_rank']:.2f} "
                  f"intra={emb_stats['intra_class_cosine']:.3f} inter={emb_stats['inter_class_cosine']:.3f}")

        if stale >= args.patience:
            print(f"[t034] Early stopping @ epoch {epoch} (patience={args.patience})")
            break

    # ─── Save training history ─────────────────────────────────────────────
    with (out_dir / "train_history.json").open("w") as f:
        json.dump({
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
            "best_emb_stats": best_emb_stats,
            "config": {
                "experiment": args.experiment,
                "seed": args.seed,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "warmup": args.warmup,
                "label_smoothing": args.label_smoothing,
                "grad_clip": args.grad_clip,
                "dropout": args.dropout,
                "onnx_init": onnx_path,
                "use_augmentation": use_aug,
                "use_contrastive": use_contrastive,
                "temperature": args.temperature,
                "contrastive_weight": contrastive_weight,
            },
            "split": {
                "train_subjects": train_ids,
                "val_subjects": val_ids,
                "test_subjects": test_ids,
                "train_trials": int(X_train.shape[0]),
                "val_trials": int(X_val.shape[0]),
                "test_trials": int(X_test.shape[0]),
            },
        }, f, indent=2)

    print(f"\n[t034] Best val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")
    print(f"[t034] Test acc (at best epoch) = {best_test_acc:.4f}")
    print(f"[t034] Best epoch {best_epoch} — effective_rank={best_emb_stats['effective_rank']:.2f}, "
          f"intra={best_emb_stats['intra_class_cosine']:.3f}, inter={best_emb_stats['inter_class_cosine']:.3f}")

    # ─── Export to ONNX ────────────────────────────────────────────────────
    if args.export_onnx and not args.no_export_onnx:
        print(f"\n[t034] Exporting to ONNX...")
        onnx_path_out = str(out_dir / "eegconformer.onnx")
        export_to_onnx(model, str(ckpt_path), onnx_path_out, device)
        print(f"[t034] Done: {out_dir}")


if __name__ == "__main__":
    main()
