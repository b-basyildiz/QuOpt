JAX Optimal Control
===================

This is an independent JAX implementation of ../Optimal_Control. The original
PyTorch files and environment are unchanged. It contains no runtime PyTorch
dependency and does not read the PyTorch Data directory.

ControlFlow.py retains the original 23 positional arguments, model construction,
gate-time/seed sweep, warm-start behavior, output paths, and CSV column layout.
QOC.bash retains the current settings: 40 segments, continuous sin^2 pulses,
cross talk and leakage, CFME4, h=0.0005, and SGD. It defaults to local execution.
Running the complete QOC.bash launches the configured 10 times x 50 seeds with
5,000 iterations per run. The tests below use small runs instead.

Implementation
--------------
ML.py uses float64 parameters and complex128 matrices. It precomputes fixed
Hamiltonian operators, pulse envelopes, rotating phases, and integration grids.
The propagation and automatic differentiation use compiled JAX scans. Nesterov
SGD and Adam updates are compiled; the scalar learning-rate scheduler runs on
the host. There is a single synchronized training step per iteration.

The CFME4 grid retains the exact per-pulse floor-plus-remainder convention and
matrix ordering of the original. Fidelity uses the equivalent complete-basis
trace identity. The return value retains the original float32 infidelity
rounding and best-iteration selection; diagnostics retain double precision.
Only the best weights need to be retained during ordinary training.

Supported training options include continuous and square pulses, CTL and
ordinary models, leakage penalty, static or time-dependent coupling, CFME4,
RK2, SRK2, SGD, ADAM, and CosineLR. Historical solver formulas are preserved;
parity with the original is not an independent accuracy guarantee for every
solver/configuration. CFME4 is the tested production configuration.

Two existing nonproduction conventions are intentionally preserved: the
ordinary square-crosstalk term has a detached gradient, and CTL's -1 drive cap
uses its original cosine formula. Neither is selected by the supplied QOC.bash.
CosineLR also preserves the original scheduler.step(infidelity) call, which
interprets infidelity as an epoch instead of advancing a cosine schedule by
iteration. The supplied QOC uses SGD with ReduceLROnPlateau.
The helper port fixes two construction errors: ragged genEvec arrays are held
as a tuple, and AnalyNeg uses NumPy sqrt for a scalar.

Random initialization
---------------------
The same numeric seed gives different initial angles in JAX and PyTorch.
Training comparisons must use the same arrays, not merely the same seed.
ML.fidelity_ml accepts the additional keyword initial_weights for this purpose.
For ControlFlow.py, set QUTRIT_INITIAL_WEIGHTS to a .npy file containing the
initial angle array. This explicit override takes precedence over warm-start
CSV loading; unset it to continue from a saved warm start. Normal JAX runs use
JAX's seeded float64 generator. The existing CSV warm-start mechanism works
across backends when supplied with the same file.

Local setup and checks
----------------------
From this directory, use a Python 3.10, 3.11, or 3.12 interpreter:

    bash setup_env.bash /path/to/python3
    source .venv/bin/activate
    python -m unittest test_jax_control -v

The project's .venv is separate from the PyTorch environment and ignored by git.
All direct and transitive runtime dependencies are pinned in requirements.txt.
The local validation uses CPU JAX/JAXlib 0.4.38 and Python 3.10; this version was
selected to work with the existing NumPy 1.26 stack and Caltech's Python 3.11.

Run the numerical benchmark with the existing PyTorch interpreter as its driver:

    /path/to/pytorch/bin/python benchmark_parity.py \
        --jax-python "$PWD/.venv/bin/python" --iterations 5

Alternate optimizer/solver comparisons (also use the PyTorch driver):

    /path/to/pytorch/bin/python test_additional_modes.py \
        --jax-python "$PWD/.venv/bin/python"

Run the real CLI/file-output comparison with the JAX interpreter:

    .venv/bin/python smoke_cli.py --torch-python /path/to/pytorch/bin/python

The benchmark compares identical stored initial arrays, full evolution
matrices, fidelities, objective gradients, each parameter update, learning
rates, and returned best weights. It includes the current QOC at normalized
times 0.1, 0.5, and 1.0 plus smaller alternative-model cases. The first JAX
iteration includes compilation and is reported separately from warmed steps.
Workers run sequentially on CPU. Both training timers include diagnostics;
thread settings are recorded, but do not imply identical backend threading.

The CLI test uses temporary directories and checks fresh outputs, repeat writes,
warm-start creation, continuation, and finalization without writing production
data. Validation artifacts and measured results are saved under validation/.
These short comparisons do not establish identical 5,000-iteration trajectories
or equal performance on Caltech hardware.
See validation/results.txt for the saved accuracy and timing summary. The
observed speedup compares this compiled/refactored JAX code with the existing
eager PyTorch implementation, not an equally optimized PyTorch rewrite.

Caltech setup
-------------
Upload this directory excluding .venv, __pycache__, and generated Data. Create
the environment on the HPC rather than copying the Mac environment:

    # After logging in, deactivate any other virtual environment first.
    cd /path/to/QuOpt/JAX_Optimal_Control
    bash setup_caltech.bash
    sbatch --time=00:10:00 --mem=2G HPC.slurm --check

setup_caltech.bash loads python/3.11.6-gcc-13.2.0-fh6i4o3. The default environment
is this directory's .venv. QUTRIT_JAX_ENV can select another JAX-specific path;
QUTRIT_JAX_PYTHON_MODULE can override the module. The existing QUTRIT_ENV is not
used, so the other control-flow environments remain separate.

After the checks and an appropriate resource pilot, set HPC="True" in QOC.bash
to submit its sweep with HPC.slurm. This launcher currently requests CPU jobs;
GPU performance has not been benchmarked.
