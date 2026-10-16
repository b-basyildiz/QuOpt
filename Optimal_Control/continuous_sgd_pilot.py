"""Short, matched-start SGD pilot through the actual ML.fidelity_ml code.

This measures early optimization, gradients, and runtime, not convergence of a
5000-iteration run. Final best controls are checked independently with DOP853.
Run one process per h with BLAS/OMP threads set to one; no production data is
written. See --help for the default durations and iteration counts.
"""

import argparse
import gc
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import ML
from continuous_step_sweep import CTLModel, provenance, read_config, reference
from helperFuncs import genDrive


def run_case(config, initial, normalized_time, iterations, h):
    model = CTLModel(config, initial[None])
    duration = normalized_time * model.base_time
    drives = [genDrive(4, l, axis) for l in (1, 2) for axis in ("x", "y")]
    leakage = [genDrive(4, 3, axis) for axis in ("x", "y")]
    anharm = np.diag([0, 0, config["anharmonicity"], 2 * config["anharmonicity"]])
    rows = []
    first_gradient = None
    original_step = torch.optim.SGD.step
    original_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau.step
    started = previous = time.perf_counter()

    def record_step(optimizer, closure=None):
        nonlocal first_gradient
        parameter = optimizer.param_groups[0]["params"][0]
        gradient = parameter.grad.detach().numpy().copy()
        if first_gradient is None:
            first_gradient = gradient
        rows.append({"iteration": len(rows) + 1,
                     "gradient_norm": float(np.linalg.norm(gradient)),
                     "learning_rate": optimizer.param_groups[0]["lr"]})
        return original_step(optimizer, closure)

    def record_scheduler(scheduler, metrics, epoch=None):
        nonlocal previous
        result = original_scheduler(scheduler, metrics, epoch)
        now = time.perf_counter()
        rows[-1].update(fidelity=1 - float(metrics.detach()), seconds=now - previous)
        previous = now
        print(json.dumps({"h": h, "normalized_time": normalized_time, **rows[-1]}), flush=True)
        return result

    with patch.object(ML.torch, "rand", return_value=torch.tensor(initial / (2 * np.pi))), \
            patch.object(torch.optim.SGD, "step", record_step), \
            patch.object(torch.optim.lr_scheduler.ReduceLROnPlateau, "step", record_scheduler):
        reported_fidelity, best_weights = ML.fidelity_ml(
            model.M, model.target, duration, iterations, 1, model.H0,
            [drives, leakage, anharm], config["maxDriveStrength"], True,
            "False", "True", h, .5, config["anharmonicity"], config["staggering"],
            "CFME4", "True", "SGD", "False", False)
    elapsed = time.perf_counter() - started
    best_model = CTLModel(config, best_weights[None])
    ref_u, _ = reference(best_model, duration, 1e-12)
    reference_fidelity = float(best_model.fidelity(ref_u)[0])
    best_index = int(np.argmin(np.array([1 - row["fidelity"] for row in rows], dtype=np.float32)))
    best_fidelity = rows[best_index]["fidelity"]
    return ({"h": h, "normalized_time": normalized_time, "duration": duration,
             "iterations": iterations, "elapsed_seconds": elapsed,
             "median_iteration_seconds": float(np.median([row["seconds"] for row in rows])),
             "reported_fidelity_float32": reported_fidelity,
             "best_fidelity": best_fidelity, "best_iteration": best_index + 1,
             "reference_fidelity": reference_fidelity,
             "best_control_absolute_fidelity_error": abs(best_fidelity - reference_fidelity),
             "history": rows}, first_gradient, best_weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("QOC.bash"))
    parser.add_argument("--controls", type=Path,
                        default=Path("validation/continuous_steps_2026-10-12/controls.npz"))
    parser.add_argument("--control-index", type=int, default=0)
    parser.add_argument("--h", type=float, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[.1, .5, 1.])
    parser.add_argument("--iterations", type=int, nargs="+", default=[12, 3, 1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.times) != len(args.iterations) or min(args.iterations) < 1 or args.h <= 0:
        parser.error("Use positive h and matching times/positive iteration counts.")
    torch.set_num_threads(1)
    config = read_config(args.config)
    if config["optimizer"] != "SGD" or config["minimizeLeakage"] != "False":
        parser.error("Pilot expects SGD with minimizeLeakage=False.")
    saved = np.load(args.controls)
    initial = saved["weights"][args.control_index]
    args.output.mkdir(parents=True, exist_ok=True)
    records, arrays = [], {"initial_weights": initial}
    metadata = {"config": config, "control_index": args.control_index,
                "controls_source": str(args.controls.resolve()),
                "reference_tolerance": 1e-12, "provenance": provenance(),
                "scope": "Actual production SGD; early iterations only, not a convergence guarantee."}
    for normalized_time, iterations in zip(args.times, args.iterations):
        record, gradient, best_weights = run_case(config, initial, normalized_time, iterations, args.h)
        records.append(record)
        key = f"t{normalized_time:.4f}"
        arrays[f"first_gradient_{key}"] = gradient
        arrays[f"best_weights_{key}"] = best_weights
        (args.output / f"h{args.h:g}.json").write_text(json.dumps({**metadata, "cases": records}, indent=2) + "\n")
        np.savez_compressed(args.output / f"h{args.h:g}.npz", **arrays)
        gc.collect()


if __name__ == "__main__":
    main()
