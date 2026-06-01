#!/bin/bash

# Usage:
# ./clean_requirements.sh requirements.txt

INPUT_FILE="${1:-requirements.txt}"
OUTPUT_FILE="requirements_clean.txt"

echo "Cleaning $INPUT_FILE ..."

python3 <<PY
import re

input_file = "$INPUT_FILE"
output_file = "$OUTPUT_FILE"

clean_lines = []

with open(input_file, "r") as f:
    for line in f:
        line = line.strip()

        # skip empty lines/comments
        if not line or line.startswith("#"):
            continue

        # remove local file installs like:
        # package @ file:///...
        if " @ file://" in line:
            pkg = line.split(" @ file://")[0].strip()
            clean_lines.append(pkg)
        else:
            clean_lines.append(line)

# remove duplicates while preserving order
seen = set()
final_lines = []

for line in clean_lines:
    if line not in seen:
        final_lines.append(line)
        seen.add(line)

with open(output_file, "w") as f:
    for line in final_lines:
        f.write(line + "\\n")

print(f"Wrote cleaned requirements to {output_file}")
print(f"Total packages: {len(final_lines)}")
PY

echo ""
echo "Done."
echo "Install with:"
echo "python -m pip install -r $OUTPUT_FILE"