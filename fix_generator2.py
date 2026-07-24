import nbformat as nbf
import os
import json

# Create a new notebook
nb = nbf.v4.new_notebook()

# List of cells
cells = []

# Helper function to create a code cell from a string
def code_cell(source):
    return nbf.v4.new_code_cell(source)

# Helper function to create a markdown cell from a string
def markdown_cell(source):
    return nbf.v4.new_markdown_cell(source)

# Cell 1: Markdown - Title and description
cell1 = '''
# EEGConformer vs PCA Evaluation Notebook

This notebook evaluates the existing EEGConformer ONNX model against the PCA baseline in Neuro-Fabric.

**Steps:**
1. Load and preprocess BCI-IV-2a data (using the same pipeline as Neuro-Fabric).
2. Load the pre-trained EEGConformer ONNX model.
3. Compute EEGConformer embeddings for the embeddings via ONNX Runtime.
4. Compute PCA embeddings (fit on training data, transform on test).
5. Compare the two methods on classification performance, embedding quality, inference latency, and more.
'''
cells.append(markdown_cell(cell1))

# Cell 2: Install dependencies (if needed)
cell2 = '''
# Install required packages if not already installed
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages
packages = [
    "numpy",
    "scipy",
    "scikit-learn",
    "onnxruntime",
    "moabb",
    "pandas",
    "matplotlib",
    "seaborn",
    "nbformat"
]

for package in packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        install(package)
'''
cells.append(code_cell(cell2))

# Cell 3: Imports and configuration loading
cell3 = '''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import onnxruntime as ort
import moabb
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

# Set seeds for reproducibility
np.random.seed(42)

# Paths
REPO_ROOT = os.getcwd()
CONFIG_PATH = os.path.join(REPO_ROOT, "training", "configs", "eegconformer-bciiv2a.yaml")
MODEL_PATH = os.path.join(REPO_ROOT, "public", "models", "eegconformer.onnx")

# Load configuration (simplified: we know the values from the config file)
# In practice, we would load the YAML, but for simplicity we hardcode the known values.
config = {
    "signal": {
        "channels": 22,
        "sample_rate_hz": 250,
        "bandpass_hz": [4, 38],
        "tmin_s": 0,
        "tmax_s": 4,  # 4 seconds
        "window_samples": 1000  # 250 Hz * 4 s
    },
    "model": {
        "n_classes": 4
    },
    "dataset": {
        "subjects": list(range(1, 10)),  # Subjects 1-9
        "holdout_subjects": [9]  # Subject 9 for hold-out test
    }
}

print("Configuration loaded:")
print(json.dumps(config, indent=2))
'''
cells.append(code_cell(cell3))

# Cell 4: Load and preprocess data (using the same preprocessing as in preprocess.py)
cell4 = '''
# We'll use the preprocess.py script to get the data in the correct format.
# Alternatively, we can replicate the steps here.

# Let's import the preprocessing function from the training scripts if possible.
import sys
sys.path.append(os.path.join(REPO_ROOT, "training", "scripts"))

# We'll import the necessary functions from preprocess.py
# Since preprocess.py is not set up as a module, we'll exec the file to get the functions.
# Alternatively, we can run the preprocess.py script and load the output.

# We choose to run the preprocess.py script to ensure we get exactly the same data.
# However, note that the preprocess.py script writes to disk. We'll run it and then load the data.

from _common import Paths, load_config

# Load config
cfg = load_config(CONFIG_PATH)
paths = Paths.from_config(cfg)

# Set environment variables for data storage
os.environ.setdefault("MNE_DATA", str(paths.cache))
os.environ.setdefault("MOABB_DATA", str(paths.cache))

# Import MOABB and paradigm
import moabb
from moabb.datasets import BNCI2014_001
from moabb
from moabb.paradigms import MotorImagery

moabb.set_log_level(")  # This is to avoid triple single quotes inside, we break it
# Actually, we cannot have triple single quotes inside. Let's avoid by using double quotes for the import.
# We'll change the import line to use double quotes.
# Let's rewrite this cell without the problematic line.
# We'll break the string and concatenate, or use double quotes for the outer string of this cell.
# Given the complexity, let's use a different approach: we'll write the cell as a list of lines and join with newline.
# But we are already using a helper function that takes a string.
# We'll define the string without triple single quotes inside.
# We can escape the single quote inside by using double quotes for the string we are having trouble with.
# The problematic line is: from moabb.paradigms import MotorImagery
# It doesn't have triple quotes, so it's safe.
# Actually, the error was because we had a triple single quote inside? Let's see: we wrote:
# from moabb.paradigms import MotorImagery
# That's fine.
# But note: we had a line: from moabb.datasets import BNCI2014_001
# That's also fine.
# The issue might be that we have a triple single quote in the string we are defining? We are using triple single quotes for the entire string, so we cannot have three single quotes in a row inside.
# Let's check the content: we don't have three single quotes in a row.
# However, we have a line that says: from moabb.paradigms import MotorImagery
# That has two single quotes in a row? No, it has none.
# So it should be safe.
# Let's try to run the script as is and see if we get a different error.
# We'll output the script and then run it.
'''
# We'll break here and rethink.
# Given the time, let's change strategy: we'll write the cell as a list of lines and then join.
# We'll do that for all cells to avoid quoting issues.
