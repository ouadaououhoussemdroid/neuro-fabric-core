import nbformat as nbf
import sys

# Read the creation script
with open('create_eval_notebook.py', 'r') as f:
    lines = f.readlines()

# Find the line where the 6th cell starts (the incomplete function)
start_of_cell6 = None
for i, line in enumerate(lines):
    if line.strip().startswith('cells.append(nbf.v4.new_code_cell("""def run_eegconformer_onnx'):
        start_of_cell6 = i
        break

if start_of_cell6 is None:
    # If not found, we'll try to find the line that starts the 6th cell by looking for the pattern
    for i, line in enumerate(lines):
        if 'cells.append(nbf.v4.new_code_cell("""def run_eegconformer_onnx' in line:
            start_of_cell6 = i
            break

# If we still didn't find it, we'll assume the 6th cell starts at line 231 (0-indexed) as per earlier grep.
if start_of_cell6 is None:
    start_of_cell6 = 230  # because line numbers in the file are 1-indexed, and we want the line at index 230 (which is the 231st line)

# Take all lines before the start of the 6th cell
relevant_lines = lines[:start_of_cell6]

# Now, we'll execute these lines to get the first 5 cells.
# We'll create a namespace and execute the relevant lines.
namespace = {'nbformat': nbf, 'nbf': nbf}
try:
    exec(''.join(relevant_lines), namespace)
    # After execution, the namespace should have a variable 'cells' which is the list of the first 5 cells.
    cells = namespace['cells']
except Exception as e:
    print(f"Failed to execute the creation script lines: {e}")
    sys.exit(1)

# Now we have the first 5 cells in the `cells` list.
# We'll create a new notebook and set its cells to these cells.
nb = nbf.v4.new_notebook()
nb['cells'] = cells

# Now we need to append the remaining cells (from our plan) to this notebook.
# We'll define the remaining cells as we did before (cells 6 to 11) and append them.

# We'll create the remaining cells using the same method as before (using lists of lines to avoid quote issues).

# We'll need the variables that are defined in the first 5 cells, such as config, ort_session, input_name, etc.
# But note: the remaining cells will be executed in the same notebook, so they will have access to the variables defined in the first 5 cells.

# Let's define the remaining cells.

# Cell 6: Function to run EEGConformer inference (completed)
cell6_lines = [
    'def run_eegconformer_onnx(X_batch):',
    '    """',
    '    Run EEGConformer inference on a batch of data.',
    '    Expects X_batch of shape [N, channels, samples] or [N, 1, channels, samples] depending on the model.',
    '    Returns embeddings of shape [N, embedding_dim].',
    '    """',
    '    # Determine if we need to add a dimension based on model\'s expected input shape',
    '    # We do this check once and cache the result using function attributes',
    '    if not hasattr(run_eegconformer_onnx, \'initialized\'):',
    '        # Run a dummy input to determine the expected input shape',
    '        dummy_input = np.random.randn(1, config[\"signal\"][\"channels\"], config[\"signal\"][\"window_samples\"]).astype(np.float32)',
    '        try:',
    '            _ = ort_session.run(None, {input_name: dummy_input})',
    '            # If successful, model expects [N, C, T]',
    '            run_eegconformer_onnx.expand_dim = False',
    '        except Exception:',
    '            # Try with an extra dimension',
    '            dummy_input2 = np.random.randn(1, 1, config[\"signal\"][\"channels\"], config[\"signal\"][\"window_samples\"]).astype(np.float32)',
    '            _ = ort_session.run(None, {input_name: dummy_input2})',
    '            # If this works, model expects [N, 1, C, T]',
    '            run_eegconformer_onnx.expand_dim = True',
    '        ',
    '        # Store other needed attributes',
    '        run_eegconformer_onnx.input_name = input_name',
    '        run_eegconformer_onnx.ort_session = ort_session',
    '        run_eegconformer_onnx.initialized = True',
    '    ',
    '    # Prepare input based on what the model expects',
    '    if run_eegconformer_onnx.expand_dim and len(X_batch.shape) == 3:',
    '        # Model expects [N, 1, C, T] but we have [N, C, T] -> add dimension at position 1',
    '        X_batch = np.expand_dims(X_batch, axis=1)',
    '    elif not run_eegconformer_onnx.expand_dim and len(X_batch.shape) == 4 and X_batch.shape[1] == 1:',
    '        # Model expects [N, C, T] but we have [N, 1, C, T] -> remove the extra dimension',
    '        X_batch = X_batch[:, 0, :, :]',
    '    # If shapes already match, do nothing',
    '    ',
    '    # Run inference',
    '    ort_inputs = {run_eegconformer_onnx.input_name: X_batch}',
    '    ort_outputs = run_eegconformer_onnx.ort_session.run(None, ort_inputs)',
    '    ',
    '    # Return the first output (assumed to be embeddings)',
    '    return ort_outputs[0]'
]
cell6_source = '\n'.join(cell6_lines)
nb['cells'].append(nbf.v4.new_code_cell(cell6_source))

# Cell 7: Compute EEGConformer embeddings for training and test data
cell7_lines = [
    '# Compute EEGConformer embeddings for training and test data',
    'print("Computing EEGConformer embeddings...")',
    '',
    '# Process in batches to avoid memory issues',
    'batch_size = 64',
    '',
    'def compute_embeddings_in_batches(X, batch_size=64):',
    '    """Compute embeddings in batches"""',
    '    n_samples = X.shape[0]',
    '    embeddings = []',
    '',
    '    for i in range(0, n_samples, batch_size):',
    '        batch_end = min(i + batch_size, n_samples)',
    '        batch = X[i:batch_end]'
    '        batch_embeddings = run_eegconformer_onnx(batch)'
    '        embeddings.append(batch_embeddings)'
    '        if i % (batch_size * 5) == 0:  # Progress update every 5 batches',
    '            print(f"  Processed {min(i+batch_size, n_samples)}/{n_samples} samples")',
    '    ',
    '    return np.vstack(embeddings)',
    '',
    '# Compute embeddings',
    'train_start = time.time()',
    'X_train_embeddings = compute_embeddings_in_batches(X_train, batch_size)',
    'train_time = time.time() - train_start',
    '',
    'test_start = time.time()',
    'X_test_embeddings = compute_embeddings_in_batches(X_test, batch_size)',
    'test_time = time.time() - test_start',
    '',
    'print(f"Training embeddings computed in {train_time:.2f}s")',
    'print(f"Test embeddings computed in {test_time:.2f}s")',
    'print(f"Training embeddings shape: {X_train_embeddings.shape}")',
    'print(f"Test embeddings shape: {X_test_embeddings.shape}")'
]
cell7_source = '\n'.join(cell7_lines)
nb['cells'].append(nbf.v4.new_code_cell(cell7_source))

# Cell 8: Compute PCA embeddings for training and test data
cell8_lines = [
    '# Compute PCA embeddings for training and test data',
    'print("Computing PCA embeddings...")',
    '',
    'def compute_pca_embeddings(X_train, X_test, n_components=32):',
    '    """',
    '    Compute PCA embeddings for train and test data.',
    '    Fits PCA on training data and transforms both train and test data.',
    '    """',
    '    # Flatten the data for PCA (from [N, C, T] to [N, C*T])',
    '    n_train, c, t = X_train.shape',
    '    n_test = X_test.shape[0]',
    '',
    '    X_train_flat = X_train.reshape(n_train, -1)',
    '    X_test_flat = X_test.reshape(n_test, -1)',
    '',
    '    # Standardize features (important for PCA)',
    '    scaler = StandardScaler()',
    '    X_train_scaled = scaler.fit_transform(X_train_flat)'
    '    X_test_scaled = scaler.transform(X_test_flat)',
    '',
    '    # Apply PCA',
    '    pca = PCA(n_components=n_components)'
    '    X_train_pca = pca.fit_transform(X_train_scaled)'
    '    X_test_pca = pca.transform(X_test_scaled)',
    '',
    '    # Also return the explained variance ratio for information',
    '    explained_variance_ratio = pca.explained_variance_ratio_',
    '',
    '    return X_train_pca, X_test_pca
