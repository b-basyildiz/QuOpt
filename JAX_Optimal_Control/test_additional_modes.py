"""Small cross-process parity checks for alternate optimizers and ODE modes.

Run this driver with the PyTorch interpreter and provide --jax-python pointing
to the separate JAX environment. Both workers execute their actual fidelity_ml;
the original code is observed with hooks, without editing its calculations.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

from benchmark_parity import (HERE, TORCH_SOURCE, compare_outputs, load_arguments,
                              make_fixture, save_worker, source_hashes)


def cases():
    common = dict(M=2, level=4, leakage=True, crossTalk="True", ContPulse="True",
                  minLeak="False", maxDriveStrength=4, anharmVal=14., stag=17.,
                  h=.002, alpha=.5, ode="CFME4", coupling="SpeedUp", iterations=3,
                  seed=7, normalized_time=.02, optimizer="SGD")
    return [dict(common, name="adam_ctl", optimizer="ADAM"),
            dict(common, name="cosine_ctl", optimizer="CosineLR"),
            dict(common, name="rk2_ctl", ode="RK2"),
            dict(common, name="srk2_ctl", ode="SRK2"),
            dict(common, name="conth_continuous", level=3, leakage=False,
                 crossTalk="False", time_dependent_coupling=True),
            dict(common, name="conth_square", level=3, leakage=False,
                 crossTalk="False", ContPulse="False", time_dependent_coupling=True),
            dict(common, name="square_crosstalk", level=3, leakage=False, ContPulse="False"),
            dict(common, name="ctl_leakage_penalty", minLeak="True")]


def arguments_for(spec_path):
    import numpy as np
    spec, initial, arguments = load_arguments(spec_path)
    arguments = list(arguments)
    arguments[17] = spec["optimizer"]
    if spec.get("time_dependent_coupling"):
        from helperFuncs import HC
        # Construct precisely genEvec's energy values, avoiding its original
        # ragged np.array constructor, which NumPy 1.26 rejects before training.
        base = 500/3
        other = base-spec["stag"]
        first = [0, base] + [(i+1)*base+i*spec["anharmVal"] for i in range(1, spec["level"])]
        second = [spec["stag"], other] + [(i+1)*other+i*spec["anharmVal"] for i in range(1, spec["level"])]
        energies = (np.asarray(first), np.asarray(second),
                    np.asarray([base+other, first[2]+second[2]]))
        arguments[5] = lambda t: HC(t, energies, spec["level"]-1)
    return spec, initial, arguments


def torch_worker(spec_path, output):
    import numpy as np
    import torch
    sys.path.insert(0, str(TORCH_SOURCE))
    import ML
    torch.set_num_threads(1)
    spec, initial, arguments = arguments_for(spec_path)
    diagnostics = {key: [] for key in (
        "unitaries", "weights_before", "weights_after", "gradients", "fidelities",
        "costs", "learning_rates_before", "learning_rates_after", "iteration_seconds")}
    optimizer_type = torch.optim.Adam if spec["optimizer"] == "ADAM" else torch.optim.SGD
    scheduler_type = (torch.optim.lr_scheduler.CosineAnnealingLR if spec["optimizer"] == "CosineLR"
                      else torch.optim.lr_scheduler.ReduceLROnPlateau)
    original_basis = ML.genTwoQuditBasis
    original_step = optimizer_type.step
    original_scheduler = scheduler_type.step
    metrics = []
    previous = time.perf_counter()

    def basis_hook(*args, **kwargs):
        frame = inspect.currentframe().f_back
        diagnostics["unitaries"].append(frame.f_locals["U_Exp"].detach().numpy().copy())
        diagnostics["weights_before"].append(frame.f_locals["R"].detach().numpy().copy())
        del frame
        return original_basis(*args, **kwargs)

    def step_hook(optimizer, closure=None):
        parameter = optimizer.param_groups[0]["params"][0]
        frame = inspect.currentframe().f_back
        while frame is not None and frame.f_code.co_name != "fidelity_ml":
            frame = frame.f_back
        if frame is None:
            raise RuntimeError("Could not inspect original fidelity_ml")
        metrics.append(float(frame.f_locals["infidelity"].detach()))
        diagnostics["costs"].append(float(frame.f_locals[
            "cost" if spec["minLeak"] == "True" else "infidelity"].detach()))
        del frame
        diagnostics["gradients"].append(parameter.grad.detach().numpy().copy())
        diagnostics["learning_rates_before"].append(optimizer.param_groups[0]["lr"])
        result = original_step(optimizer, closure)
        diagnostics["weights_after"].append(parameter.detach().numpy().copy())
        return result

    def scheduler_hook(scheduler, *args, **kwargs):
        nonlocal previous
        result = original_scheduler(scheduler, *args, **kwargs)
        # CosineAnnealingLR also calls step() once inside its constructor.
        if len(metrics) > len(diagnostics["fidelities"]):
            diagnostics["fidelities"].append(1-metrics[-1])
            diagnostics["learning_rates_after"].append(scheduler.optimizer.param_groups[0]["lr"])
            now = time.perf_counter()
            diagnostics["iteration_seconds"].append(now-previous)
            previous = now
        return result

    started = time.perf_counter()
    with patch.object(ML.torch, "rand", return_value=torch.tensor(initial/(2*np.pi))), \
            patch.object(ML, "genTwoQuditBasis", basis_hook), \
            patch.object(optimizer_type, "step", step_hook), \
            patch.object(scheduler_type, "step", scheduler_hook):
        result = ML.fidelity_ml(*arguments)
    save_worker(output, diagnostics, result, {
        "framework": "torch", "version": torch.__version__, "python": sys.version,
        "platform": platform.platform(), "spec": spec, "total_seconds": time.perf_counter()-started})


def jax_worker(spec_path, output):
    import jax
    sys.path.insert(0, str(HERE))
    import ML
    spec, initial, arguments = arguments_for(spec_path)
    started = time.perf_counter()
    result, diagnostics = ML.fidelity_ml(*arguments, initial_weights=initial, return_diagnostics=True)
    save_worker(output, diagnostics, result, {
        "framework": "jax", "version": jax.__version__, "python": sys.version,
        "platform": platform.platform(), "spec": spec, "total_seconds": time.perf_counter()-started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-python", default=sys.executable)
    parser.add_argument("--jax-python", default=str(HERE/".venv/bin/python"))
    parser.add_argument("--output", type=Path, default=HERE/"validation/additional_modes")
    parser.add_argument("--worker", choices=("torch", "jax"), help=argparse.SUPPRESS)
    parser.add_argument("--case-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    if args.worker:
        {"torch": torch_worker, "jax": jax_worker}[args.worker](args.case_file, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg",
                       TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1", JAX_PLATFORMS="cpu",
                       MPLCONFIGDIR=str(args.output.resolve()/"mplconfig"))
    environment.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    report = {"scope": "Three matched updates per alternate optimizer/model; correctness smoke checks.",
              "notes": "ContH supplies original energy values directly: original genEvec ragged-array "
                       "construction fails with modern NumPy before training. CosineLR preserves the "
                       "original scheduler.step(infidelity) metric-as-epoch behavior.",
              "source_sha256": source_hashes(), "cases": []}
    report["source_sha256"][str(Path(__file__).resolve().relative_to(HERE.parent))] = (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    selected = cases()
    if args.cases:
        unknown = set(args.cases)-{case["name"] for case in selected}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        selected = [case for case in selected if case["name"] in args.cases]
    for spec in selected:
        name = spec["name"]
        print(f"Checking {name}", flush=True)
        spec_path = args.output/f"{name}.fixture.json"
        spec_path.write_text(json.dumps(spec, indent=2)+"\n")
        make_fixture(spec, spec_path.with_suffix(".npz"))
        outputs = {}
        for backend, executable in (("torch", args.torch_python), ("jax", args.jax_python)):
            output = args.output/f"{name}.{backend}.result"
            command = [executable, str(Path(__file__).resolve()), "--worker", backend,
                       "--case-file", str(spec_path.resolve()), "--output", str(output.resolve())]
            with (args.output/f"{name}.{backend}.log").open("w") as log:
                subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
            outputs[backend] = output.with_suffix(".npz")
        comparison = compare_outputs(outputs["torch"], outputs["jax"])
        report["cases"].append({"name": name, "spec": spec, "comparison": comparison})
        report["passed"] = all(case["comparison"]["passed"] for case in report["cases"])
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(f"  parity={comparison['passed']}", flush=True)
    if not report["passed"]:
        raise SystemExit("Parity failed; inspect report.json")


if __name__ == "__main__":
    main()
