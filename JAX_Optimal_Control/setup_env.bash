#!/bin/bash
# Usage: bash setup_env.bash [path/to/python3]
set -euo pipefail
cd "$(dirname "$0")"
base_python="${1:-python3}"
export QUTRIT_JAX_ENV="${QUTRIT_JAX_ENV:-$PWD/.venv}"
export PYTHONNOUSERSITE=1
"$base_python" - <<'PY'
import sys
if sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12)):
    raise SystemExit("These pinned dependencies require Python 3.10, 3.11, or 3.12.")
print("Creating the JAX environment with", sys.executable)
PY
"$base_python" -m venv "$QUTRIT_JAX_ENV"
"$QUTRIT_JAX_ENV/bin/python" -m pip install --only-binary=:all: -r requirements.txt
"$QUTRIT_JAX_ENV/bin/python" -m pip check
printf 'Activate with: source "%s/bin/activate"\n' "$QUTRIT_JAX_ENV"
