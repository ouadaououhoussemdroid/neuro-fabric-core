# Read the file
with open('training/notebooks/eegconformer_pca_evaluation.ipynb', 'r') as f:
    lines = f.readlines()

# Line numbers are 0-indexed.
# We want to replace lines 204 to 211 inclusive (which correspond to lines 205-212 in 1-indexed)
# with the correct block.

# The line at index 203 is the elif condition line (1-indexed line 204) which we want to keep.
# So we start replacing at index 204.

new_block = [
    '        embedding_idx = 1\n',
    '        logits_idx = 0\n',
    '    else:\n',
    '        # Default to first output if neither matches expected embedding dim\n',
    '        embedding_idx = 0\n',
    '        logits_idx = 1 if len(dummy_output) > 1 else None\n'
]

# Replace lines 204 to 211 (inclusive) with new_block
# We are removing 8 lines and putting in 6 lines.
lines[204:212] = new_block

# Write back
with open('training/notebooks/eegconformer_pca_evaluation.ipynb', 'w') as f:
    f.writelines(lines)

print("Fixed the notebook by replacing lines 205-212 (1-indexed).")
