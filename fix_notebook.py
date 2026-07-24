import nbformat as nbf

# Create notebook
nb = nbf.v4.new_notebook()
cells = []

def code_cell(lines):
    return nbf.v4.new_code_cell('\n'.join(lines))

def markdown_cell(lines):
    return nbf.v4.new_markdown_cell('\n'.join(lines))

# Cell 0: Title
cells.append(markdown_cell([
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
]))

# Cell 1: Install dependencies
cells.append(code_cell([
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
    "    \"pandas\",",
    "    \"scikit-learn\",",
    "    \"onnxruntime\",",
    "    \"moabb\",\n",
    "    \"pandas\",\n",
    "    \"matplotlib\",\n",
    "    \"seaborn\",\n",
    "    \"nbformat\"\n",
    "]",
    "",
    "for package in packages:",
    "    try:",
    "        __import__(package)",
    "    except ImportError:",
    "        print(f\"Installing {package}...\")",
    "        install(package)"
]))

# Cell 2: Imports and config
cells.append(code_cell([
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
    "# Load configuration",
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
]))

# Cell 3: Data loading (using _common.py)
cells.append(code_cell([
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
]))

# Cell 4: Load ONNX model
cells.append(code_cell([
    "# Load the ONNX model",
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
    "input_shape = ort_session.get_inputs()[0].shape",
    "print(f\"Expected input shape: {input_shape}\")",
    "",
    "# We expect the input to be [batch, channels, samples] (as per the export script)",
    "# Let's run a dummy input to see the output shapes and names.",
    "dummy_input = np.random.randn(1, config[\"signal\"][\"channels\"], config[\"signal\"][\"window_samples\"]).astype(np.float32)",
    "dummy_output = ort_session.run(None, {input_name: dummy_input})",
    "print(f\"Dummy output shapes: {[out.shape for out in dummy_output]}\")",
    "print(f\"Output names corresponding to shapes: {list(zip(output_names, [out.shape for out in dummy_output]))}\")",
    "",
    "# Determine which output is the embedding and which is the logits.",
    "# From the export script, the first output is embedding, second is logits.",
    "# But we can also check the shapes: embedding should be [1, 32], logits [1, 4]",
    "embedding_idx = 0",
    "logits_idx = 1",
    "if len(dummy_output) >= 2:",
    "    if dummy_output[0].shape[1] == config.get(\"embedding_dim\", 32):",
    "        embedding_idx = 0",
    "        logits_idx = 1",
    "    elif dummy_output[1].shape[1] == config.get(\"embedding_dim\", 32):",
    "        embedding_idx = 1",
    "        logits_idx = 0",
    "    else:",
    "        # Default to first output if neither matches expected embedding dim",
    "        embedding_idx = 0",
    "        logits_idx = 1 if len(dummy_output) > 1 else None",
    "",
    "print(f\"Embedding output index: {embedding_idx}\")",
    "if logits_idx is not None:",
    "    print(f\"Logits output index: {logits_idx}\")"
]))

# Cell 5: EEGConformer inference function
cells.append(code_cell([
    "def run_eegconformer_onnx(X_batch):",
    "    \"\"\"",
    "    Function to run EEGConformer inference on a batch of data.",
    "    Determines if we need to add a dimension based on model's expected input shape",
    "    \"\"\"",
    "    # Determine if we need to add a dimension based on model's expected input shape",
    "    input_shape = ort_session.get_inputs()[0].shape",
    "    # If the model expects a 4D input [batch, 1, channels, samples], we need to add a dimension",
    "    if len(input_shape) == 4 and input_shape[1] == 1:",
    "        # Add a dimension at position 1 (after batch)",
    "        X_batch = np.expand_dims(X_batch, axis=1)",
    "    # Run the model",
    "    outputs = ort_session.run(None, {input_name: X_batch})",
    "    # Return the embedding (using the embedding_idx we determined earlier)",
    "    return outputs[embedding_idx]"
]))

# Cell 6: Compute EEGConformer embeddings
cells.append(code_cell([
    "# Compute EEGConformer embeddings for train and test data",
    "import time",
    "",
    "print(\"Computing EEGConformer embeddings...\")",
    "start_time = time.time()",
    "X_train_emb = run_eegconformer_onnx(X_train)",
    "X_test_emb = run_eegconformer_onnx(X_test)",
    "eegconformer_time = time.time() - start_time",
    "",
    "print(f\"EEGConformer embeddings computed in {eegconformer_time:.2f}s\")",
    "print(f\"Training embeddings shape: {X_train_emb.shape}\")",
    "print(f\"Test embeddings shape: {X_test_emb.shape}\")"
]))

# Cell 7: PCA baseline
cells.append(code_cell([
    "# Compute PCA embeddings (fit on training data, transform on test)",
    "from sklearn.decomposition import PCA",
    "from sklearn.preprocessing import StandardScaler",
    "import numpy as np",
    "import time",
    "",
    "print(\"Computing PCA embeddings...\")",
    "pca_start = time.time()",
    "",
    "# Flatten the data for PCA (from [N, C, T] to [N, C*T])",
    "n_train, c, t = X_train.shape",
    "n_test = X_test.shape[0]",
    "X_train_flat = X_train.reshape(n_train, -1)"
    "X_test_flat = X_test.reshape(n_test, -1)",
    "",
    "# Standardize features (important for PCA)",
    "scaler = StandardScaler()",
    "X_train_scaled = scaler.fit_transform(X_train_flat)"
    "X_test_scaled = scaler.transform(X_test_flat)",
    "",
    "# Apply PCA",
    "n_components = X_train_emb.shape[1]  # same as EEGConformer embedding dimension"
    "pca = PCA(n_components=n_components)"
    "X_train_pca = pca.fit_transform(X_train_scaled)"
    "X_test_pca = pca.transform(X_test_scaled)",
    "",
    "pca_time = time.time() - pca_start",
    "",
    "print(f\"PCA embeddings computed in {pca_time:.2f}s\")",
    "print(f\"Training PCA embeddings shape: {X_train_pca.shape}\")",
    "print(f\"Test PCA embeddings shape: {X_test_pca.shape}\")",
    "print(f\"Explained variance ratio (first 5 components): {pca.explained_variance_ratio_[:5]}\")",
    "print(f\"Total explained variance ({n_components} components): {np.sum(pca.explained_variance_ratio_):.3f}\")"
]))

# Cell 8: Evaluation
cells.append(code_cell([
    "# Train classifiers and evaluate performance",
    "from sklearn.linear_model import LogisticRegression",
    "from sklearn.metrics import accuracy_score",
    "import time",
    "",
    "print(\"Training classifiers and evaluating performance...\")",
    "",
    "# EEGConformer embeddings",
    "clf_ec = LogisticRegression(max_iter=1000, random_state=42)",
    "clf_ec.fit(X_train_emb, y_train)",
    "y_pred_ec = clf_ec.predict(X_test_emb)",
    "acc_ec = accuracy_score(y_test, y_pred_ec)",
    "print(f\"EEGConformer + Logistic Regression accuracy: {acc_ec:.4f}\")",
    "",
    "# PCA embeddings",
    "clf_pca = LogisticRegression(max_iter=1000, random_state=42)",
    "clf_pca.fit(X_train_pca, y_train)",
    "y_pred_pa = clf_pca.predict(X_test_pca)",
    "acc_pa = accuracy_score(y_test, y_pred_pa)",
    "print(f\"PCA + Logistic Regression accuracy: {acc_pa:.4f}\")",
    "",
    "# Compare the two approaches",
    "print(\"\\nComparison:\")",
    "print(f\"EEGConformer accuracy: {acc_ec:.4f}\")",
    "print(f\"PCA accuracy:          {acc_pa:.4f}\")",
    "print(f\"Difference (EEGConformer - PCA): {acc_ec - acc_pa:.4f}\")",
    "",
    "# Inference latency (per sample) - we already have the total time for embeddings",
    "# We can compute the average time per sample for EEGConformer",
    "eegconformer_latency_per_sample = eegconformer_time / len(X_train)  # training time per sample",
    "p(X_train)  # training time per sample",
    "pca_latency_per_sample = pca_time / len(X_train)  # PCA fitting time per sample (note: PCA fitting is done once)",
    "",
    "print(f\"EEGConformer latency per sample (training): {eegconformer_latency_per_sample*1000:.2f} ms\")",
    "print(f\"PCA latency per sample (fitting): {pca_latency_per_sample*1000:.2f} ms\")",
    "",
    "# Store results for later use",
    "eval_results = {",
    "    \"eegconformer_accuracy\": float(acc_ec),",
    "    \"pca_accuracy\": float(acc_pa),",
    "    \"eegconformer_latency_per_sample_ms\": float(eegconformer_latency_per_sample * 1000),",
    "    \"pca_latency_per_sample_ms\": float(pca_latency_per_sample * 1000),",
    "    \"training_samples\": int(len(X_train)),",
    "    \"test_samples\": int(len(X_test)),",
    "    \"embedding_dimension\": int(X_train_emb.shape[1]),",
    "}"
]))

# Cell 9: Save report
cells.append(code_cell([
    "# Save results to a JSON file",
    "import json",
    "from pathlib import Path",
    "",
    "# Define the output directory",
    "output_dir = Path(REPO_ROOT) / \"training\" / \"artefacts\" / \"eegconformer-bciiv2a-v1\"",
    "output_dir.mkdir(parents=True, exist_ok=True)",
    "",
    "output_file = output_dir / \"colab_evaluation_report.json\"",
    "",
    "# Prepare the final report",
    "report = {",
    "    \"repository\": str(REPO_ROOT),",
    "    \"configuration\": {",
    "        \"channels\": CHANNELS,",
    "        \"sample_rate_hz\": SAMPLE_RATE,"
    "        \"window_samples\": WINDOW_SAMPLES,"
    "        \"embedding_dimension\": X_train_emb.shape[1],"
    "        \"n_classes\": N_CLASSES,"
    "    },",
    "    \"results\": eval_results,"
    "}",
    "",
    "# Write the JSON file",
    "with open(output_file, 'w') as f:",
    "    json.dump(report, f, indent=2)",
    "",
    "print(f\"Evaluation report saved to: {output_file}\")"
]))

# Assign cells and write notebook
nb['cells'] = cells
output_path = "training/notebooks/eegconformer_pca_evaluation.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook written to {output_path}")