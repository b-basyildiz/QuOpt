#!/bin/bash
# Caltech CPU template. Submit from the directory containing ControlFlow.py.
# Export QUTRIT_ENV="$HOME/QuOpt/.venv" first.
# These are initial resource requests; size memory/time from a pilot's seff report.
#SBATCH --job-name=QuOpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=quopt-%j.out
#SBATCH --error=quopt-%j.err
#SBATCH --export=ALL

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch, not bash.}"
: "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory.}"
: "${QUTRIT_ENV:?Export the absolute path to your qutrit virtual environment before sbatch.}"
cd "$SLURM_SUBMIT_DIR"

if [[ ! -f ControlFlow.py || ! -f cfme.py ]]; then
    echo "Submit from the uploaded project directory containing ControlFlow.py and cfme.py." >&2
    exit 1
fi
if [[ ! -x "$QUTRIT_ENV/bin/python" ]]; then
    echo "Python environment not found: $QUTRIT_ENV/bin/python" >&2
    exit 1
fi

# Use the same module when creating the venv and running the job. This default
# is the module confirmed available on bbasyild's Caltech account.
module load "${QUTRIT_PYTHON_MODULE:-python/3.11.6-gcc-13.2.0-fh6i4o3}"
source "$QUTRIT_ENV/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/quopt-mpl-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "Job $SLURM_JOB_ID on $(hostname); working directory: $PWD"
echo "Python: $QUTRIT_ENV/bin/python; CPU threads: $OMP_NUM_THREADS"

if [[ $# -eq 1 && $1 == --check ]]; then
    exec srun "$QUTRIT_ENV/bin/python" -u -m unittest test_bugfixes test_continuous_pulses -v
fi
if [[ $# -eq 1 && $1 == --smoke ]]; then
    exec srun "$QUTRIT_ENV/bin/python" -u smoke_caltech.py
fi
if [[ $# -ne 23 ]]; then
    echo "Expected 23 ControlFlow.py arguments, --check, or --smoke." >&2
    exit 2
fi

exec srun "$QUTRIT_ENV/bin/python" -u ControlFlow.py "$@"
