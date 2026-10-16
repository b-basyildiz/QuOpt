#!/bin/bash
# Run on Caltech after deactivating any other virtual environment.
set -euo pipefail
cd "$(dirname "$0")"
export QUTRIT_JAX_PYTHON_MODULE="${QUTRIT_JAX_PYTHON_MODULE:-python/3.11.6-gcc-13.2.0-fh6i4o3}"
module load "$QUTRIT_JAX_PYTHON_MODULE"
python3 - <<'PY'
import sys
if sys.prefix != sys.base_prefix:
    raise SystemExit("Deactivate the current venv before running setup_caltech.bash.")
if sys.version_info[:2] != (3, 11):
    raise SystemExit("Expected the Caltech Python 3.11 module.")
PY
exec bash setup_env.bash "$(command -v python3)"
