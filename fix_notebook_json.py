import json

# Read the notebook as JSON
with open('training/notebooks/eegconformer_pca_evaluation.ipynb', 'r') as f:
    nb = json.load(f)

# We are going to fix the source of the cell that contains the problematic lines.
# We know the problematic lines are in a code cell. Let's find the cell that has the string "if len(dummy_output) >= 2:" in its source.
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        # Check if any line in the source contains the string
I realize: we have to the control character at position 15 column 13.

 We directly fix by replacing the problematic string the problematic part with the correct one.

 Let's use a Python script to load the notebook, find the cell with the problematic string, and fix it<tool_call>
<function=Bash>
<parameter=command>
cat > fix_notebook.py << 'EOF'
import json

# Read the notebook
with open('training/notebooks/eegconformer_pca_evaluation.ipynb', 'r') as f:
    nb = json.load(f)

# We'll iterate through the cells to find the one that contains the problematic string.
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Join the source lines into a single string for easier replacement
        source = ''.join(cell['source'])
        # We are looking for the problematic part: the lines after "elif dummy_output[1].shape[1] == config.get(\"embedding_dim\", 32):\n"
        # We know the problematic part starts with a line that has spaces and then a newline, then some lines, then a stray line.
        # Instead, we'll replace the entire problematic block with the correct block.
        # We'll look for the pattern: 
        #   '    elif dummy_output[1].shape[1] == config.get(\"embedding_dim\", 32):\n' followed by some broken lines.
        # We'll replace from that line until the line before the next cell or until we see a line that starts with '    \"' (indicating the next string in the list) or the end of the cell.
        # But note: the source is a list of strings, each string is a line (including newline).
        # We'll work with the list.

        new_source = []
        i = 0
        while i < len(cell['source']):
            line = cell['source'][i]
            # Check if this line is the start of the problematic elif block
            if line.strip() == '    elif dummy_output[1].shape[1] == config.get(\"embedding_dim\", 32):\n':
                # We found the start of the elif block.
                # We want to keep this line, then replace the next lines until we see a line that is not part of the broken block.
                new_source.append(line)  # keep the elif line
                i += 1
                # Now, we expect the next lines to be the broken ones. We'll skip them and put the correct ones.
                # Skip until we find a line that is not part of the broken block (i.e., a line that is a string in the list, not the broken lines we see)
                # But note: the broken block in the file is:
                #   Line 205: '        \n'   (which is a string with spaces and newline)
                #   Line 206: '        embedding_idx = 1\n'
                #   Line 207: '        logits_idx = 0\n'
                #   Line 208: '   \n'       (which is a string with spaces and newline)
                #   Line 209: '        \"        embedding_idx = 1\n"'  (which is a string that starts with spaces and a quote)
                #   ... and so on.
                # We want to replace from the current i (which is the line after the elif) until we see a line that is a string that starts with spaces and a quote (which is the next string in the list) or until we have processed enough.
                # Instead, we'll just replace the next 8 lines (from i to i+7) with the correct block, but we don't know the exact length.
                # Let's look at the intended correct block for the elif and else:
                #   "        embedding_idx = 1\n",
                #   "        logits_idx = 0\n",
                #   "    else:\n",
                #   "        # Default to first output if neither matches expected embedding dim\n",
                #   "        embedding_idx = 0\n",
                #   "        logits_idx = 1 if len(dummy_output) > 1 else None\n",
                # But note: the elif line we just kept is the one that ends with colon and newline.
                # So after the elif line, we want:
                #   "        embedding_idx = 1\n",
                #   "        logits_idx = 0\n",
                #   "    else:\n",
                #   "        # Default to first output if neither matches expected embedding dim\n",
                #   "        embedding_idx = 0\n",
                #   "        logits_idx = 1 if len(dummy_output) > 1 else None\n",
                # However, note that in the original (correct) structure, the elif block is:
                #   "    elif dummy_output[1].shape[1] == config.get(\"embedding_dim\", 32):\n",
                #   "        embedding_idx = 1\n",
                #   "        logits_idx = 0\n",
                #   "    else:\n",
                #   "        # Default to first output if neither matches expected embedding dim\n",
                #   "        embedding_idx = 0\n",
                #   "        logits_idx = 1 if len(dummy_output) > 1 else None\n",
                # So we will replace the next lines (starting at i) with the 5 lines above.

                # But note: the current line at i (after the elif) is the broken line 205: '        \n'
                # We want to replace from i to the end of the broken block. We don't know the exact length, but we know we want to put 5 lines.
                # However, we must be careful not to remove too much or too little.

                # Let's instead replace the next lines until we see a line that is a string that starts with '    \"' (which is the next string in the list) or until we have processed 10 lines (to be safe).
                # We'll collect the next lines until we see a line that is a string that starts with '    \"' (and note that the string in the list includes the newline and the quote at the end? Actually, each line in the source list is a string that ends with newline, and the string content includes the quote if it's a string line).
                # In the source list, a line that is a string in the notebook (like the ones we see in the cell source) is a string that starts and ends with double quotes and contains the escaped content, and ends with \n\"? Actually, no: each element in the source list is a string that is the content of the line (including the newline) but without the outer quotes? Let me think: when we save the notebook, each line in the source is a string that is exactly what is between the quotes in the JSON, and it includes the newline as a character (so the string ends with \n? Actually, in the JSON, the string is written with the newline as \n, so when we load it, the string contains the actual newline character?).

                # Actually, in the notebook JSON, the source is a list of strings, each string is the line content (including the newline character at the end, but note that the newline is represented as \n in the JSON string, so when loaded, it is a single newline character).

                # So, in the source list, a line that is a string in the notebook (like the ones that are the content of the cell) is a string that does NOT start and end with quotes. The quotes are in the JSON representation.

                # Therefore, in the source list, we can distinguish between:
                #   - A line that is a string in the notebook: it does not start with a space? Actually, it can start with any character.
                #   - But note: the broken lines we see in the source list (from the cat -A output) are:
                #         Line 205: '        \n'   -> this is a
