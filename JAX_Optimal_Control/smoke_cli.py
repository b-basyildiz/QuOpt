"""Compare both real ControlFlow.py entry points, including CSV and warm starts.

Run with the JAX environment. The PyTorch environment is supplied separately;
neither environment is modified. All training output goes into temporary folders.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
TORCH_DIR = HERE.parent / "Optimal_Control"


def run(command, cwd, env):
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stdout}\n{result.stderr}")
    return result


def compare_outputs(torch_dir, jax_dir):
    files = sorted(p.relative_to(torch_dir) for p in torch_dir.rglob("*.csv"))
    jax_files = sorted(p.relative_to(jax_dir) for p in jax_dir.rglob("*.csv"))
    if files != jax_files or not files:
        raise AssertionError(f"CSV paths differ: {files} vs {jax_files}")
    errors = {}
    for relative in files:
        left, right = torch_dir / relative, jax_dir / relative
        weights = "Weights" in relative.parts
        if weights:
            if left.read_text().splitlines()[0] != right.read_text().splitlines()[0]:
                raise AssertionError(f"Weight headers differ: {relative}")
        a = np.loadtxt(left, delimiter=",", skiprows=int(weights), ndmin=2)
        b = np.loadtxt(right, delimiter=",", skiprows=int(weights), ndmin=2)
        if a.shape != b.shape:
            raise AssertionError(f"Output shape differs for {relative}: {a.shape} vs {b.shape}")
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-8, err_msg=str(relative))
        errors[str(relative)] = float(np.max(np.abs(a-b)))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-python", required=True)
    parser.add_argument("--jax-python", default=sys.executable)
    parser.add_argument("--output", type=Path, default=HERE / "validation" / "cli_smoke.json")
    args = parser.parse_args()
    args.torch_python = str(Path(args.torch_python).expanduser().absolute())
    args.jax_python = str(Path(args.jax_python).expanduser().absolute())
    stages = []
    with tempfile.TemporaryDirectory(prefix="quopt-jax-cli-") as scratch:
        root = Path(scratch)
        torch_dir, jax_dir = root / "torch", root / "jax"
        torch_dir.mkdir()
        jax_dir.mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   JAX_ENABLE_X64="True", JAX_PLATFORMS="cpu",
                   TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
                   XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
                   MPLCONFIGDIR=str(root / "mpl"))
        env.pop("QUTRIT_INITIAL_WEIGHTS", None)
        initial = root / "initial.npy"
        run([args.torch_python, "-c",
             "import numpy as np, torch, sys; torch.manual_seed(1); "
             "np.save(sys.argv[1], (torch.rand((40,8),dtype=torch.double)*2*np.pi).numpy())",
             str(initial)], root, env)
        # Exact current continuous CTL setup, deterministic seed, three SGD updates.
        common = ["iSWAP", "3", "SpeedUp", "40", "1", "14", "True", "17",
                  "CFME4", "0.0005", "0.5", "True", "True", "False", "40",
                  "0.1", "0.1", "-1", "3", "SGD", "0", "0"]
        for name, warm_start, shared_initial in (
            ("fresh", -1, True), ("repeat_existing_csv", -1, True),
            ("warm_start_create", 0, True), ("warm_start_continue", 0, False),
            ("warm_start_final", 1, False),
        ):
            run([args.torch_python, str(TORCH_DIR / "ControlFlow.py"), *common, str(warm_start)], torch_dir, env)
            jax_env = dict(env)
            if shared_initial:
                jax_env["QUTRIT_INITIAL_WEIGHTS"] = str(initial)
            run([args.jax_python, str(HERE / "ControlFlow.py"), *common, str(warm_start)], jax_dir, jax_env)
            errors = compare_outputs(torch_dir, jax_dir)
            stages.append({"stage": name, "max_absolute_errors": errors, "passed": True})
            print(name, "passed; max error", max(errors.values()), flush=True)
        report = {"passed": True, "iterations_per_call": 3, "configuration": common,
                  "initial_weights_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
                  "torch_python": args.torch_python, "jax_python": args.jax_python,
                  "absolute_tolerance": 1e-8, "relative_tolerance": 1e-8, "stages": stages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
