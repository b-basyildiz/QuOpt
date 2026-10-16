"""Compare the actual PyTorch and JAX trainers in isolated Python processes.

Run this driver with the existing PyTorch Python, passing --jax-python for the
separate JAX environment. Both workers consume identical saved inputs. No
production Data directory is written. Timings include synchronized training
iterations and diagnostics, with the first JAX iteration reported separately.
This is an early-iteration parity check, not a 5,000-step convergence test.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
TORCH_SOURCE = HERE.parent / "Optimal_Control"


def case_specs(iterations=3):
    common = dict(M=40, level=4, leakage=True, crossTalk="True",
                  ContPulse="True", minLeak="False", maxDriveStrength=40,
                  anharmVal=14., stag=17., h=.0005, alpha=.5, ode="CFME4",
                  coupling="SpeedUp", iterations=iterations, seed=1)
    cases = [dict(common, name=f"qoc_t{t:g}", normalized_time=t)
             for t in (.1, .5, 1.)]
    cases.extend([
        dict(common, name="ctl_square", M=3, normalized_time=.1,
             h=.01, ContPulse="False", maxDriveStrength=4),
        dict(common, name="leakage_continuous", M=3, normalized_time=.1,
             h=.01, crossTalk="False", maxDriveStrength=4),
        dict(common, name="leakage_penalty", M=3, normalized_time=.1,
             h=.01, crossTalk="False", maxDriveStrength=4, minLeak="True"),
        dict(common, name="qubit_square", M=3, level=2, leakage=False,
             normalized_time=.1, h=.01, crossTalk="False",
             ContPulse="False", maxDriveStrength=4, coupling="XX"),
        dict(common, name="qutrit_crosstalk", M=3, level=3, leakage=False,
             normalized_time=.1, h=.01, maxDriveStrength=4),
    ])
    return cases


def make_fixture(spec, path):
    """Use the original pipeline's constructors and RNG as the input oracle."""
    import numpy as np
    import torch
    sys.path.insert(0, str(TORCH_SOURCE))
    from helperFuncs import gateGen, genCouplMat, genDrive
    level = spec["level"]
    upper = level - 1 if spec["leakage"] else level
    drives = np.array([genDrive(level, transition, axis)
                       for transition in range(1, upper)
                       for axis in ("x", "y")])
    leakage_drives = np.array([genDrive(level, level - 1, axis)
                               for axis in ("x", "y")])
    anharm = np.zeros((level, level))
    # Match ControlFlow.py's non-CTL leakage drift exactly.
    anharm_values = np.array([spec["anharmVal"] * (i + 1)
                             for i in range(level - 2)])
    for index in range(2, level):
        anharm[-(index - 1), -(index - 1)] = anharm_values[::-1][index - 2]
    columns = 8 if level >= 4 and spec["leakage"] and spec["crossTalk"] == "True" else len(drives) * 2
    torch.manual_seed(spec["seed"])
    initial = (torch.rand((spec["M"], columns), dtype=torch.double) * 2 * np.pi).numpy()
    np.savez_compressed(path, initial=initial, drives=drives,
                        leakage_drives=leakage_drives, anharm=anharm,
                        target=gateGen("iSWAP", level, upper),
                        H0=genCouplMat(spec["coupling"], level))


def load_arguments(spec_path):
    import numpy as np
    spec = json.loads(Path(spec_path).read_text())
    fixture = np.load(Path(spec_path).with_suffix(".npz"))
    drives = list(fixture["drives"])
    if spec["leakage"]:
        drives = [drives, list(fixture["leakage_drives"]), fixture["anharm"]]
    arguments = (spec["M"], fixture["target"], spec["normalized_time"] * np.pi / 2,
                 spec["iterations"], spec["seed"], fixture["H0"], drives,
                 spec["maxDriveStrength"], spec["leakage"], spec["minLeak"],
                 spec["crossTalk"], spec["h"], spec["alpha"], spec["anharmVal"],
                 spec["stag"], spec["ode"], spec["ContPulse"], "SGD", "False", False)
    return spec, fixture["initial"], arguments


def torch_worker(spec_path, output):
    import inspect
    from unittest.mock import patch
    import numpy as np
    import torch
    sys.path.insert(0, str(TORCH_SOURCE))
    import ML
    torch.set_num_threads(1)
    spec, initial, arguments = load_arguments(spec_path)
    diagnostics = {key: [] for key in (
        "unitaries", "weights_before", "weights_after", "gradients",
        "fidelities", "costs", "learning_rates_before", "learning_rates_after",
        "iteration_seconds")}
    original_basis = ML.genTwoQuditBasis
    original_step = torch.optim.SGD.step
    original_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau.step
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
        # PyTorch decorates optimizer.step with a profiling wrapper.
        while frame is not None and frame.f_code.co_name != "fidelity_ml":
            frame = frame.f_back
        if frame is None:
            raise RuntimeError("Could not locate the original fidelity_ml frame")
        cost = frame.f_locals["cost" if spec["minLeak"] == "True" else "infidelity"]
        diagnostics["costs"].append(float(cost.detach()))
        del frame
        diagnostics["gradients"].append(parameter.grad.detach().numpy().copy())
        diagnostics["learning_rates_before"].append(optimizer.param_groups[0]["lr"])
        result = original_step(optimizer, closure)
        diagnostics["weights_after"].append(parameter.detach().numpy().copy())
        return result

    def scheduler_hook(scheduler, metrics, epoch=None):
        nonlocal previous
        result = original_scheduler(scheduler, metrics, epoch)
        diagnostics["fidelities"].append(1 - float(metrics.detach()))
        diagnostics["learning_rates_after"].append(scheduler.optimizer.param_groups[0]["lr"])
        now = time.perf_counter()
        diagnostics["iteration_seconds"].append(now - previous)
        previous = now
        return result

    started = time.perf_counter()
    with patch.object(ML.torch, "rand", return_value=torch.tensor(initial / (2 * np.pi))), \
            patch.object(ML, "genTwoQuditBasis", basis_hook), \
            patch.object(torch.optim.SGD, "step", step_hook), \
            patch.object(torch.optim.lr_scheduler.ReduceLROnPlateau, "step", scheduler_hook):
        result = ML.fidelity_ml(*arguments)
    elapsed = time.perf_counter() - started
    save_worker(output, diagnostics, result, {
        "framework": "torch", "version": torch.__version__,
        "python": sys.version, "platform": platform.platform(),
        "device": "cpu", "torch_threads": torch.get_num_threads(),
        "total_seconds": elapsed, "spec": spec})


def jax_worker(spec_path, output):
    import jax
    sys.path.insert(0, str(HERE))
    import ML
    spec, initial, arguments = load_arguments(spec_path)
    started = time.perf_counter()
    result, diagnostics = ML.fidelity_ml(*arguments, initial_weights=initial,
                                         return_diagnostics=True)
    elapsed = time.perf_counter() - started
    save_worker(output, diagnostics, result, {
        "framework": "jax", "version": jax.__version__,
        "python": sys.version, "platform": platform.platform(),
        "devices": [str(device) for device in jax.devices()],
        "x64_enabled": jax.config.x64_enabled,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "total_seconds": elapsed, "spec": spec})


def save_worker(output, diagnostics, result, metadata):
    import numpy as np
    output = Path(output)
    arrays = {key: np.asarray(diagnostics[key]) for key in (
        "unitaries", "weights_before", "weights_after", "gradients", "fidelities",
        "costs", "learning_rates_before", "learning_rates_after", "iteration_seconds")}
    arrays["reported_fidelity"] = np.asarray(result[0])
    arrays["best_weights"] = np.asarray(result[1])
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    times = arrays["iteration_seconds"]
    metadata.update(first_iteration_seconds=float(times[0]),
                    warmed_iteration_seconds=[float(value) for value in times[1:]],
                    median_warmed_iteration_seconds=float(np.median(times[1:])) if len(times) > 1 else None,
                    reported_fidelity=float(result[0]))
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def compare_outputs(torch_output, jax_output):
    import numpy as np
    baseline = np.load(torch_output)
    candidate = np.load(jax_output)
    tolerances = {"unitaries": 2e-8, "weights_before": 2e-8,
                  "weights_after": 2e-8, "gradients": 2e-8,
                  "fidelities": 2e-9, "costs": 2e-9,
                  "learning_rates_before": 1e-14, "learning_rates_after": 1e-14,
                  "reported_fidelity": 1e-7, "best_weights": 2e-8}
    result = {}
    for key, tolerance in tolerances.items():
        first, second = baseline[key], candidate[key]
        shapes_match = first.shape == second.shape
        finite = bool(np.all(np.isfinite(first)) and np.all(np.isfinite(second)))
        error = float(np.max(np.abs(first - second))) if shapes_match else float("inf")
        result[key] = {"max_absolute_error": error, "absolute_tolerance": tolerance,
                       "passed": shapes_match and finite and error <= tolerance}
    result["passed"] = all(value["passed"] for value in result.values())
    return result


def source_hashes():
    paths = [TORCH_SOURCE / name for name in ("ML.py", "cfme.py", "helperFuncs.py", "QOC.bash")]
    paths += [HERE / name for name in ("ML.py", "cfme.py", "helperFuncs.py", "benchmark_parity.py")]
    return {str(path.relative_to(HERE.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-python", default=sys.executable)
    parser.add_argument("--jax-python", default=str(HERE / ".venv/bin/python"))
    parser.add_argument("--output", type=Path, default=HERE / "validation/parity")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--cases", nargs="+", help="Optional case-name subset")
    parser.add_argument("--worker", choices=("torch", "jax"), help=argparse.SUPPRESS)
    parser.add_argument("--case-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        {"torch": torch_worker, "jax": jax_worker}[args.worker](args.case_file, args.output)
        return
    if args.iterations < 3:
        parser.error("At least three updates are required for this smoke benchmark.")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = case_specs(args.iterations)
    if args.cases:
        unknown = set(args.cases) - {case["name"] for case in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in args.cases]
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg",
                       TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
                       MPLCONFIGDIR=str(args.output.resolve() / "mplconfig"),
                       JAX_PLATFORMS="cpu")
    environment.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    report = {"scope": f"Same-input CPU smoke comparison; {args.iterations} SGD updates per case, not full convergence.",
              "timing_notes": "Sequential workers; first iteration separated from warmed iterations. "
                  "JAX first iteration includes compilation, and all JAX iterations synchronize. "
                  "Both include diagnostic host copies. PyTorch uses one CPU thread; XLA flags and "
                  "BLAS thread limits are recorded, but are not a guarantee of identical backend threading.",
              "source_sha256": source_hashes(), "cases": []}
    for spec in cases:
        name = spec["name"]
        print(f"Comparing {name}: {spec['iterations']} updates", flush=True)
        spec_path = args.output / f"{name}.fixture.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        make_fixture(spec, spec_path.with_suffix(".npz"))
        metadata = {}
        outputs = {}
        for backend, executable in (("torch", args.torch_python), ("jax", args.jax_python)):
            output = args.output / f"{name}.{backend}"
            # suffix '.torch' / '.jax' is part of the stem, not replaced.
            output = output.with_name(output.name + ".result")
            command = [executable, str(Path(__file__).resolve()), "--worker", backend,
                       "--case-file", str(spec_path.resolve()), "--output", str(output.resolve())]
            with (args.output / f"{name}.{backend}.log").open("w") as log:
                subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
            outputs[backend] = output.with_suffix(".npz")
            metadata[backend] = json.loads(output.with_suffix(".json").read_text())
        comparison = compare_outputs(outputs["torch"], outputs["jax"])
        torch_time = metadata["torch"]["median_warmed_iteration_seconds"]
        jax_time = metadata["jax"]["median_warmed_iteration_seconds"]
        case_result = {"name": name, "spec": spec, "comparison": comparison,
                       "timing": metadata, "warmed_speed_ratio_torch_over_jax": torch_time / jax_time}
        report["cases"].append(case_result)
        report["passed"] = all(case["comparison"]["passed"] for case in report["cases"])
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"  parity={comparison['passed']}; warmed torch={torch_time:.4f}s, "
              f"jax={jax_time:.4f}s, ratio={torch_time / jax_time:.2f}", flush=True)
    if not report["passed"]:
        raise SystemExit("Parity failed; inspect report.json for the failing comparisons.")


if __name__ == "__main__":
    main()
