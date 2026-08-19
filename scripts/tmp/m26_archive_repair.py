#!/usr/bin/env python3
"""Repair benchmark_archive.json: truncate at end of valid 10-subj reassessment record,
then close JSON properly. The file was corrupted by a failed float32 append."""
import json, os

SRC_TMP = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(SRC_TMP)
REPO = os.path.dirname(SCRIPTS)
ARCHIVE_PATH = os.path.join(REPO, "reports", "benchmark_archive.json")

# Read the raw file, find the end of the m26-retrieval-reassessment record (line 4514: "},")
# Truncate everything after the 25th valid experiment, then close JSON.
with open(ARCHIVE_PATH, "r") as f:
    lines = f.readlines()

# Line 4514 (index 4513) is "    }," — closes m26-retrieval-reassessment
# We need to end the experiments array here and close the root object.
# Remove the trailing comma and add closing brackets.
keep_until = 4514  # keep lines 1..4514 (indices 0..4513)

# Line at index 4513 is "    },\n" — change trailing comma to nothing
lines[4513] = "    }\n"
# Keep lines up to and including index 4513, then close
valid_lines = lines[:4514]

with open(ARCHIVE_PATH, "w") as f:
    f.writelines(valid_lines)
    f.write("  ]\n")
    f.write("}\n")

# Validate
with open(ARCHIVE_PATH, "r") as f:
    arch = json.load(f)

print(f"Archive repaired: {len(arch['experiments'])} experiments")
for exp in arch["experiments"]:
    print(f"  - {exp['id']}")
