import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# List of cells
cells = []

# Helper function to create a code cell from a list of lines
def code_cell(lines):
    return nbf.v4.new_code_cell('\n'.join(lines))

# Helper function to create a markdown cell from a list of lines
def markdown_cell(lines):
    return nbf.v4.new_markdown_cell('\n'.join(lines))

# Cell 1: Markdown - Title and description
cell1_lines = [
    "# EEGConformer vs PCA Evaluation Notebook",
    "",
    "This notebook evaluates the existing EEGConformer ONNX model against the PCA baseline in Neuro-Fabric.",
    "",
    "**Steps:**",
    "1. Load and preprocess BCI-IV-2a data (using the same pipeline as Neuro-Fabric).",
    "2. Load the pre-trained EEGConformer ONNX model.",
    "3. Compute EEGConformer embeddings for the embeddings via ONNX Runtime.",
    "4. Compute PCA embeddings (fit on training data, transform on test).",
    "5. Compare the two methods on classification performance, embedding quality, inference latency, and more."
]
cells.append(markdown_cell(cell1_lines))

# Cell 2: Install dependencies (if needed)
cell2_lines = [
    "# Install required packages if not already installed",
    "import subprocess",
    "import sys",
    "",
    "def install(package):",
    "    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", package])",
    "",
    "# List of packages",
    "packages = [",
    "    \"numpy\",",
    "    \"scipy\",",
    "    \"scikit-learn\",",
    "    \"onnxruntime\",",
    "    \"moabb\",",
    "    \"pandas\",",
    "    \"matplotlib\",",
    "    \"seaborn\",",
    "    \"nbformat\"",
    "]",
    "",
    "for package in packages:",
    "    try:",
    "        __import__(package)",
    "    except ImportError:",
    "        print(f\"Installing {package}...\")",
    "        install(package)"
]
cells.append(code_cell(cell2_lines))

# Cell 3: Imports and configuration loading
cell3_lines = [
    "import os",
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import time",
    "import json",
    "from sklearn.decomposition import PCA",
    "from sklearn.preprocessing import StandardScaler",
    "from sklearn.metrics import accuracy_score",
    "from sklearn.pipeline import make_pipeline",
    "import onnxruntime as ort",
    "import moabb",
    "from moabb.datasets import BNCI2014_001",
    "from moabb.paradigms import MotorImagery",
    "",
    "# Set seeds for reproducibility",
    "np.random.seed(42)",
    "",
    "# Paths",
    "REPO_ROOT = os.getcwd()",
    "CONFIG_PATH = os.path.join(REPO_ROOT, \"training\", \"configs\", \"eegconformer-bciiv2a.yaml\")",
    "MODEL_PATH = os.path.join(REPO_ROOT, \"public\", \"models\", \"eegconformer.onnx\")",
    "",
    "# Load configuration (simplified: we know the values from the config file)",
    "# In practice, we would load the YAML, but for simplicity we hardcode the known values.",
    "config = {",
    "    \"signal\": {",
    "        \"channels\": 22,",
    "        \"sample_rate_hz\": 250,",
    "        \"bandpass_hz\": [4, 38],",
    "        \"tmin_s\": 0,",
    "        \"tmax_s\": 4,  # 4 seconds",
    "        \"window_samples\": 1000  # 250 Hz * 4 s",
    "    },",
    "    \"model\": {",
    "        \"n_classes\": 4",
    "    },",
    "    \"dataset\": {",
    "        \"subjects\": list(range(1, 10)),  # Subjects 1-9",
    "        \"holdout_subjects\": [9]  # Subject 9 for hold-out test",
    "    }",
    "}",
    "",
    "print(\"Configuration loaded:\")",
    "print(json.dumps(config, indent=2))"
]
cells.append(code_cell(cell3_lines))

# Cell 4: Load and preprocess data (using the same preprocessing as in preprocess.py)
cell4_lines = [
    "# We'll use the preprocess.py script to get the data in the correct format.",
    "# Alternatively, we can replicate the steps here.",
    "",
    "# Let's import the preprocessing function from the training scripts if possible.",
    "import sys",
    "sys.path.append(os.path.join(REPO_ROOT, \"training\", \"scripts\"))",
    "",
    "# We'll import the necessary functions from preprocess.py",
    "# Since preprocess.py is not set up as a module, we'll exec the file to get the functions.",
    "# Alternatively, we can run the preprocess.py script and load the output.",
    "",
    "# We choose to run the preprocess.py script to ensure we get exactly the same data.",
    "# However, note that the preprocess.py script writes to disk. We'll run it and then load the data.",
    "",
    "from _common import Paths, load_config",
    "",
    "# Load config",
    "cfg = load_config(CONFIG_PATH)",
    "paths = Paths.from_config(cfg)",
    "",
    "# Set environment variables for data storage",
    "os.environ.setdefault(\"MNE_DATA\", str(paths.cache))",
    "os.environ.setdefault(\"MOABB_DATA\", str(paths.cache))",
    "",
    "# Import MOABB and paradigm",
    "import moabb",
    "from moabb.datasets import BNCI2014_001",
    "from moabb.paradigms import MotorImagery",
    "",
    "moabb.set_log_level(\"WARNING\")",
    "",
    "sig = cfg[\"signal\"]",
    "paradigm = MotorImagery(",
    "    fmin=sig[\"bandpass_hz\"][0],",
    "    fmax=sig[\"bandpass_hz\"][1],",
    "    tmin=sig[\"tmin_s\"],",
    "    tmax=sig[\"tmax_s\"],",
    "    resample=sig[\"sample_rate_hz\"],",
    "    n_classes=cfg[\"model\"][\"n_classes\"]",
    ")",
    "",
    "dataset = BNCI2014_001()",
    "subjects = cfg[\"dataset\"][\"subjects\"]",
    "holdout = set(cfg[\"dataset\"][\"holdout_subjects\"])",
    "",
    "print(\"[preprocess] extracting epochs via MOABB MotorImagery paradigm\")",
    "X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)",
    "X = np.asarray(X, dtype=np.float32)",
    "",
    "# Contract enforcement",
    "target_T = sig[\"window_samples\"]",
    "if X.shape[-1] != target_T:",
    "    # Defensive crop / pad if MOABB returns ±1 sample due to rounding.",
    "    if X.shape[-1] > target_T:",
    "        X = X[..., :target_T]",
    "    else:",
    "        pad_width = ((0, 0), (0, 0), (0, target_T - X.shape[-1]))",
    "        X = np.pad(X, pad_width, mode=\"constant\")",
    "",
    "# Per-trial z-score per channel (matches the runtime preprocessing contract)",
    "def _zscore(x: np.ndarray) -> np.ndarray:",
    "    # x: [N, C, T] -> per-trial per-channel",
    "    mean = x.mean(axis=-1, keepdims=True)",
    "    std = x.std(axis=-1, keepdims=True) + 1e-6",
    "    return ((x - mean) / std).astype(np.float32)",
    "",
    "X = _zscore(X)",
    "y = np.asarray(labels, dtype=np.int64)",
    "subjects_arr = np.asarray(metadata[\"subject\"], dtype=np.int64)",
    "",
    "# Split into train and test based on holdout subject",
    "train_mask = np.isin(subjects_arr, list(set(subjects) - holdout))",
    "test_mask = np.isin(subjects_arr, holdout)",
    "",
    "X_train, y_train = X[train_mask], y[train_mask]",
    "X_test, y_test = X[test_mask], y[test_mask]",
    "subjects_train = subjects_arr[train_mask]",
    "subjects_test = subjects_arr[test_mask]",
    "",
    "print(f\"Training samples: {X_train.shape[0]}\")",
    "print(f\"Test samples: {X_test.shape[0]}\")",
    "print(f\"Data shape: {X_train.shape[1:]} (channels, samples)\")"
]
cells.append(code_cell(cell4_lines))

# Cell 5: Load the ONNX model and prepare for inference
cell5_lines = [
    "# Load the ONNX model and prepare for inference",
    "print(f\"Loading ONNX model from: {MODEL_PATH}\")",
    "ort_session = ort.InferenceSession(MODEL_PATH)",
    "",
    "# Get input and output details",
    "input_name = ort_session.get_inputs()[0].name",
    "output_names = [output.name for output in ort_session.get_outputs()]",
    "",
    "print(f\"Input name: {input_name}\")",
    "print(f\"Output names: {output_names}\")",
    "",
    "# Verify the input shape",
    "inpu
