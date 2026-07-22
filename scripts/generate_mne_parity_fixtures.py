"""T-006 — Generate MNE golden-file fixtures for the TS↔MNE parity harness.

Produces `training/cache/parity/mne_golden.npz` containing reference outputs
of MNE's bandpass, notch, and segmentation applied to the same synthetic
signal the TS test uses. The TS test then compares its output against these
arrays with a tight numerical tolerance.

Run (after installing MNE):
    python scripts/generate_mne_parity_fixtures.py
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "training" / "cache" / "parity"
OUT.mkdir(parents=True, exist_ok=True)


def synthetic_signal(fs: int = 250, n: int = 1000, channels: int = 4) -> np.ndarray:
    """Deterministic multi-channel test signal (matches the TS fixture)."""
    rng = np.random.default_rng(42)
    t = np.arange(n) / fs
    data = np.zeros((channels, n), dtype=np.float32)
    for ch in range(channels):
        # 10 Hz sine + 50 Hz mains + small noise (matches TS test generator).
        data[ch] = (
            np.sin(2 * np.pi * 10 * t + ch * 0.5) * 50.0
            + np.sin(2 * np.pi * 50 * t) * 5.0
            + rng.standard_normal(n).astype(np.float32) * 0.1
        )
    return data


def main() -> None:
    import mne

    mne.set_log_level("WARNING")
    fs = 250
    channels = 4
    raw_data = synthetic_signal(fs=fs, n=1000, channels=channels)
    info = mne.create_info(
        [f"ch{c}" for c in range(channels)], sfreq=fs, ch_types="eeg"
    )
    raw = mne.io.RawArray(raw_data, info)

    # Bandpass 8-30 Hz (same params as TS test).
    # TS uses biquad with Q=1/sqrt(2) for lowpass and highpass (Butterworth order 2).
    # Use IIR Butterworth order 2.
    raw_bp = raw.copy().filter(
        8,
        30,
        method="iir",
        iir_params={"order": 2, "ftype": "butter"},
    )
    bandpass_out = raw_bp.get_data()

# Notch at 50 Hz.
    # TS uses a notch with Q=30 (equivalent to bandwidth ~50/30=1.666 Hz).
    # MNE's notch_filter expects freqs and notch_widths.
    bandwidth = 50.0 / 30.0  # ~1.6667 Hz
    raw_notch = raw.copy().notch_filter(
        freqs=[50.0],
        notch_widths=bandwidth,
        method="iir",
    )
    notch_out = raw_notch.get_data()

    # Segmentation: 2 s windows, 50% overlap.
    window_samples = fs * 2
    step = window_samples // 2
    n_windows = (raw_data.shape[1] - window_samples) // step + 1
    seg_out = np.zeros((n_windows, channels, window_samples), dtype=np.float32)
    for i in range(n_windows):
        s = i * step
        seg_out[i] = raw_data[:, s : s + window_samples]

    np.savez(
        OUT / "mne_golden.npz",
        raw=raw_data,
        bandpass=bandpass_out.astype(np.float32),
        notch=notch_out.astype(np.float32),
        segments=seg_out,
        fs=fs,
        channels=channels,
        window_samples=window_samples,
        step=step,
    )
    print(f"[parity] wrote golden fixtures → {OUT / 'mne_golden.npz'}")


if __name__ == "__main__":
    main()
