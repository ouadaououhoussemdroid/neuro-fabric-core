import json

# Read the notebook
with open('training/notebooks/eegconformer_pca_evaluation.ipynb', 'r') as f:
    nb = json.load(f)

# We assume the function is in cell 6 (index 6)
cell_index = 6
if nb['cells'][cell_index]['cell_type'] != 'code':
    # Try to find the cell with the function definition
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            # Check if the first line contains the function definition
            if cell['source'] and 'def run_eegconformer_onnx' in cell['source'][0]:
                cell_index = i
                break

# Define the correct source for the function
correct_source = [
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
    "            _ = ort_session.run(None, {input_name: dummy_input})\n",
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

# Replace the cell's source
nb['cells'][cell_index]['source'] = correct_source

# Write back
with open('training/notebooks/eegconformer_pca_evaluation.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Fixed cell {cell_index}")
