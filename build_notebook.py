import nbformat as nbf
import re

# Read the creation script
with open('create_eval_notebook.py', 'r') as f:
    create_content = f.read()

# Extract the first 5 cells using regex
pattern = r'cells\.append\(nbf\.v4\.new_(markdown|code)_cell\("""([\s\S]*?)"""\)'
matches = re.findall(pattern, create_content, re.DOTALL)
first5 = matches[:5]  # list of (cell_type, content)

# Create notebook
nb = nbf.v4.new_notebook()
cells = []

# Add first 5 cells
for cell_type, content in first5:
    # Split content into lines and ensure each line ends with newline
    lines = content.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    lines = [line + '\n' for line in lines]
    if cell_type == 'markdown':
        cells.append(nbf.v4.new_markdown_cell(''.join(lines)))
    else:
        cells.append(nbf.v4.new_code_cell(''.join(lines)))

# Now define the remaining cells (6 to 11) as lists of strings

# Cell 6: Function to run EEGConformer inference
cell6 = [
    "def run_eegconformer_onnx(X_batch):\n",
    '    """\n',
    "    Run EEGConformer inference on a batch of data.\n",
    "    Expects X_batch of shape [N, channels, samples] or [N, 1, channels, samples] depending on the model.\n",
    "    Returns embeddings of shape [N, embedding_dim].\n",
    '    """\n',
    "    # Determine if we need to add a dimension based on model's expected input shape\n",
    "    # We do this check once and cache the result using function attributes\n",
    "    if not hasattr(run_eegconformer_onnx, 'initialized'):\n",
    "        # Run a dummy input to determine the expected input shape\n",
    "        dummy_input = np.random.randn(1, config[\"signal\"][\"channels\"], config[\"signal\"][\"window_samples\"]).astype(np.float32)\n",
    "        try:\n",
    ",
    "            # If successful, model expects [N, C, T]\n",
    "            run_eegconformer_onnx.expand_dim = False\n",
    "        except Exception:\n",
    "            # Try with an extra dimension\n",
    "            dummy_input2 = np.random.randn(1, 1, config[\"signal\"][\"channels\"], config[\"signal\"][\"window_samples\"]).astype(np.float32)\n",
    "            _ = ort_session.run(None, {input_name: dummy_input2})\n",
    "            # If this works, model expects [N, 1, C, T]\n",
    "            run_eegconformer_onnx.expand_dim = True\n",
    "        \n",
    "        # Store other needed attributes\n",
    "        run_eegconformer_onnx.input_name = input_name\n",
    "        run_eegconformer_onnx.ort_session = ort_session\n",
    "        run_eegconformer_onnx.initialized = True\n",
    "    \n",
    "    # Prepare input based on what the model expects\n",
    "    if run_eegconformer_onnx.expand_dim and len(X_batch.shape) == 3:\n",
    "        # Model expects [N, 1, C, T] but we have [N, C, T] -> add dimension at position 1\n",
    "        X_batch = np.expand_dims(X_batch, axis=1)\n",
    "    elif not run_eegconformer_onnx.expand_dim and len(X_batch.shape) == 4 and X_batch.shape[1] == 1:\n",
    "        # Model expects [N, C, T] but we have [N, 1, C, T] -> remove the extra dimension\n",
    "        X_batch = X_batch[:, 0, :, :]\n",
    "    # If shapes already match, do nothing\n",
    "    \n",
    "    # Run inference\n",
    "    ort_inputs = {run_eegconformer_onnx.input_name: X_batch}\n",
    "    ort_outputs = run_eegconformer_onnx.ort_session.run(None, ort_inputs)\n",
    "    \n",
    "    # Return the first output (assumed to be embeddings)\n",
    "    return ort_outputs[0]\n"
]
cells.append(nbf.v4.new_code_cell(''.join(cell6)))

# Cell 7: Compute EEGConformer embeddings for training and test data
cell7 = [
    "# Compute EEGConformer embeddings for training and test data\n",
    "print(\"Computing EEGConformer embeddings...\")\n",
    "\n",
    "# Process in batches to avoid memory issues\n",
    "batch_size = 64\n",
    "\n",
    "def compute_embeddings_in_batches(X, batch_size=64):\n",
    '    """Compute embeddings in batches"""',
    "\n",
    "    n_samples = X.shape[0]\n",
    "    embeddings = []\n",
    "\n",
    "    for i in range(0, n_samples, batch_size):\n",
    "        batch_end = min(i + batch_size, n_samples)\n",
    "        batch = X[i:batch_end]\n",
    "        batch_embeddings = run_eegconformer_onnx(batch)\n",
    "        embeddings.append(batch_embeddings)\n",
    "        if i % (batch_size * 5) == 0:  # Progress update every 5 batches\n",
    "            print(f\"  Processed {min(i+batch_size, n_samples)}/{n_samples} samples\")\n",
    "    \n",
    "    return np.vstack(embeddings)\n",
    "\n",
    "# Compute embeddings\n",
    "train_start = time.time()\n",
    "X_train_embeddings = compute_embeddings_in_batches(X_train, batch_size)\n",
    "train_time = time.time() - train_start\n",
    "\n",
    "test_start = time.time()\n",
    "X_test_embeddings = compute_embeddings_in_batches(X_test, batch_size)\n",
    "test_time = time.time() - test_start\n",
    "\n",
    "print(f\"Training embeddings computed in {train_time:.2f}s\")\n",
    "print(f\"Test embeddings computed in {test_time:.2f}s\")\n",
    "print(f\"Training embeddings shape: {X_train_embeddings.shape}\")\n",
    "print(f\"Test embeddings shape: {X_test_embeddings.shape}\")\n"
]
cells.append(nbf.v4.new_code_cell(''.join(cell7)))

# Cell 8: Compute PCA embeddings for training and test data
cell8 = [
    "# Compute PCA embeddings for training and test data\n",
    "print(\"Computing PCA embeddings...\")\n",
    "\n",
    "def compute_pca_embeddings(X_train, X_test, n_components=32):\n",
    '    """\n',
    "    Compute PCA embeddings for train and test data.\n",
    "    Fits PCA on training data and transforms both train and test data.\n",
    '    """\n',
    "\n",
    "    # Flatten the data for PCA (from [N, C, T] to [N, C*T])\n",
    "    n_train, c, t = X_train.shape\n",
    "    n_test = X_test.shape[0]\n",
    "\n",
    "    X_train_flat = X_train.reshape(n_train, -1)\n",
    "    X_test_flat = X_test.reshape(n_test, -1)\n",
    "\n",
    "    # Standardize features (important for PCA)\n",
    "    scaler = StandardScaler()\n",
    "    X_train_scaled = scaler.fit_transform(X_train_flat)\n",
    "    X_test_scaled = scaler.transform(X_test_flat)\n",
    "\n",
    "    # Apply PCA\n",
    "    pca = PCA(n_components=n_components)\n",
    "    X_train_pca = pca.fit_transform(X_train_scaled)\n",
    "    X_test_pca = pca.transform(X_test_scaled)\n",
    "\n",
    "    # Also return the explained variance ratio for information\n",
    "    explained_variance_ratio = pca.explained_variance_ratio_\n",
    "\n",
    "    return X_train_pca, X_test_pca, explained_variance_ratio, scaler, pca\n",
    "\n",
    "pca_start = time.time()\n",
    "X_train_pca, X_test_pca, explained_variance_ratio, scaler, pca = compute_pca_embeddings(\n",
    "    X_train, X_test, n_components=32\n",
    ")\n",
    "pca_time = time.time() - pca_start\n",
    "\n",
    "print(f\"PCA embeddings computed in {pca_time:.2f}s\")\n",
    "print(f\"Training PCA embeddings shape: {X_train_pca.shape}\")\n",
    "print(f\"Test PCA embeddings shape: {X_test_pca.shape}\")\n",
    "print(f\"Explained variance ratio (first 5 components): {explained_variance_ratio[:5]}\")\n",
    "print(f\"Total explained variance (32 components): {np.sum(explained_variance_ratio):.3f}\")\n"
]
cells.append(nbf.v4.new_code_cell(''.join(cell8)))

# Cell 9: Train classifiers and evaluate performance
cell9 = [
    "# Train classifiers and evaluate performance\n",
    "print(\"Training classifiers and evaluating performance...\")\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "\n",
    "# Function to evaluate a classifier\n",
    "def evaluate_classifier(X_train, y_train, X_test, y_test, model_name=\"Classifier\"):\n",
    "    # Train logistic regression\n
