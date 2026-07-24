# Read the file
with open('training/notebooks/eegconformer_pca_evaluation.ipynb', 'r') as f:
    lines = f.readlines()

# Line numbers are 0-indexed.
# We want to replace lines 208 to 211 (inclusive) with two new lines.
# Because:
#   line 208 (0 (11? Wait, let's map:
#   1-indexed line 209 -> index 208
#   1-indexed line 210 -> index 209
#   1-indexed line 211 -> index 210
#   1-indexed line 212 -> index 211
# We want to replace indices 208, 209, 210, 211 with two lines.

new_lines = [
    '    "        embedding_idx = 1\n",\n',
    '    "        logits_idx = 0\n",\n'
]

# Replace lines 208 to 211 (inclusive) with new_lines
lines[208:212] = new_lines

# Write back
with open('training/notebooks/eegconformer_pca_evaluation.ipynb', 'w') as f:
    f.writelines(lines)

print("Fixed the notebook.")
