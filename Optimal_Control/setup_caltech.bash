#!/bin/bash
# Run with bash from the uploaded QuOpt directory. Does not submit jobs.
set -euo pipefail
cd "$(dirname "$0")"

export QUTRIT_PYTHON_MODULE=python/3.11.6-gcc-13.2.0-fh6i4o3
export QUTRIT_ENV="$HOME/QuOpt/.venv"
export PYTHONNOUSERSITE=1

module load "$QUTRIT_PYTHON_MODULE"
python3 - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Expected Python 3.11 after module load; got {sys.version}.")
if sys.prefix != sys.base_prefix:
    raise SystemExit("Another venv is active. Run deactivate, then repeat bash setup_caltech.bash.")
print("Base Python:", sys.executable, flush=True)
PY

if [[ -x "$QUTRIT_ENV/bin/python" ]]; then
    "$QUTRIT_ENV/bin/python" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit("The existing ~/QuOpt/.venv uses a different Python version; stopping without replacing it.")
PY
fi

echo "Creating/completing $QUTRIT_ENV (this can be silent for a while)..."
python3 -m venv --prompt QuOpt "$QUTRIT_ENV"
"$QUTRIT_ENV/bin/python" -m pip install --upgrade pip
"$QUTRIT_ENV/bin/python" -m pip install --only-binary=:all: \
    -r requirements-caltech.txt 'torch==1.13.1+cpu' \
    --extra-index-url https://download.pytorch.org/whl/cpu
"$QUTRIT_ENV/bin/python" -m pip check
echo "QuOpt environment installed. Next: submit HPC.slurm --smoke."
