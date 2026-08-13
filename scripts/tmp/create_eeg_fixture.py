"""Create a realistic EEG fixture file for end-to-end testing."""
import numpy as np
import os

# 22 channels, 2500 samples (10 seconds at 250 Hz)
channels = 22
samples = 2500
sample_rate = 250

data = np.zeros((channels, samples), dtype=np.float32)

for c in range(channels):
    t = np.arange(samples, dtype=np.float64) / sample_rate
    # Alpha wave (10 Hz), beta (20 Hz), gamma (40 Hz) + noise
    alpha = np.sin(2 * np.pi * 10 * t) * 20.0
    beta = np.sin(2 * np.pi * 20 * t) * 10.0
    gamma = np.sin(2 * np.pi * 40 * t) * 5.0
    noise = np.random.RandomState(42 + c).normal(0, 2, samples).astype(np.float32)
    modulation = np.sin(2 * np.pi * 0.5 * t + c * 0.3) * 3.0
    data[c] = (alpha + beta + gamma + noise + modulation).astype(np.float32)

# Save as .npy
out_dir = "test-fixtures"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "eeg-raw-sample.npy")
np.save(out_path, data)
print(f"[fixture] Created {out_path}: shape={data.shape}, dtype={data.dtype}")
print(f"[fixture] Sample values (channel 0, first 5): {data[0, :5]}")
