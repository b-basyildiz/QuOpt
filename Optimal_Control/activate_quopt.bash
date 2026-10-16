#!/bin/bash
# Source this file so activation persists in your current terminal:
#   source ~/QuOpt/activate_quopt.bash
# Do not set shell options here: they would also affect the calling shell.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Activate in your current shell with: source ~/QuOpt/activate_quopt.bash" >&2
    exit 1
fi

if [[ ! -r "$HOME/QuOpt/.venv/bin/activate" || ! -x "$HOME/QuOpt/.venv/bin/python" ]]; then
    echo "QuOpt environment is missing. Run bash ~/QuOpt/setup_caltech.bash first." >&2
    return 1
fi
if ! type module >/dev/null 2>&1; then
    echo "The module command is unavailable. Source this script on the HPC after logging in." >&2
    return 1
fi

if declare -F deactivate >/dev/null; then
    deactivate || return 1
fi

export QUTRIT_PYTHON_MODULE=python/3.11.6-gcc-13.2.0-fh6i4o3
export QUTRIT_ENV="$HOME/QuOpt/.venv"
export PYTHONNOUSERSITE=1

module load "$QUTRIT_PYTHON_MODULE" || return 1
source "$QUTRIT_ENV/bin/activate" || return 1
cd "$HOME/QuOpt" || return 1
printf 'QuOpt active: %s\n' "$QUTRIT_ENV"
