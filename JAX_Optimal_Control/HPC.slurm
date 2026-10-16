#!/bin/bash
# CPU launcher. Submit from JAX_Optimal_Control after setup_caltech.bash.
#SBATCH --job-name=QuOpt-JAX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=quopt-jax-%j.out
#SBATCH --error=quopt-jax-%j.err
#SBATCH --export=ALL

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch.}"
: "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory.}"
cd "$SLURM_SUBMIT_DIR"
export QUTRIT_JAX_ENV="${QUTRIT_JAX_ENV:-$PWD/.venv}"
module load "${QUTRIT_JAX_PYTHON_MODULE:-python/3.11.6-gcc-13.2.0-fh6i4o3}"
if [[ ! -f ControlFlow.py || ! -f ML.py || ! -x "$QUTRIT_JAX_ENV/bin/python" ]]; then
    echo "Submit from JAX_Optimal_Control after installing its separate JAX environment." >&2
    exit 1
fi
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TF_NUM_INTRAOP_THREADS="$OMP_NUM_THREADS"
export TF_NUM_INTEROP_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$OMP_NUM_THREADS}"

echo "JAX job $SLURM_JOB_ID on $(hostname); Python: $QUTRIT_JAX_ENV/bin/python"
if [[ $# -eq 1 && $1 == --check ]]; then
    exec srun "$QUTRIT_JAX_ENV/bin/python" -u -m unittest test_jax_control -v
fi
if [[ $# -ne 23 ]]; then
    echo "Expected 23 ControlFlow.py arguments or --check." >&2
    exit 2
fi
exec srun "$QUTRIT_JAX_ENV/bin/python" -u ControlFlow.py "$@"
