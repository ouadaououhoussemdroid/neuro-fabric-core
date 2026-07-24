# Read the file
with open('training/notebooks/eegconformer_pca_evaluation.ipynb', 'r') as f:
    lines = f.readlines()

# We are going to replace lines 204 to 211 (0-indexed) because:
#   line 205 (1-indexed) -> index 204
#   line 212 (1-indexed) -> index 211
# We want to replace 8 lines (from index 204 to 211 inclusive) with 8 new lines.

# The new lines we want:
new_lines = [
    '    if dummy_output[0].shape[1] == config.get("embedding_dim", 32):\n',
    '        embedding_idx = 0\n',
    '        logits_idx = 1\n',
    '    elif dummy_output[1].shape[1] == config.get("embedding_dim", 32):\n',
    '        embedding_idx = 1\n',
    '        logits_idx = 0\n',
    '    else:\n',
    '        # Default to first output if neither matches expected embedding dim\n',
    '        embedding_idx = 0\n',
    '        logits_idx = 1 if len(dummy_output) > 1 else None\n'
]

# Replace lines 204 to 211 (inclusive) with new_lines
# Note: we are removing 8 lines and putting in 9 lines (because we split the else into two lines for clarity, but we can keep as 8 if we want).
# Actually, the original had 8 lines (205-212) and we are putting 9 lines. That's okay.

# But note: the original lines 205-212 are 8 lines. We are replacing them with 9 lines.
# Let's do:
lines[204:212] = new_lines

# Write back
with open('training/notebooks/eegconformer_pca_evaluation.ipynb', 'w') as f:
    f.writelines(lines)

print("Fixed the notebook by replacing lines 205-212.")
