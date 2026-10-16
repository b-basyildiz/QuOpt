"""Reproducible fixed-control CFME4 convergence sweep for QOC.bash's qutrit CTL model.

Run with the qutrit_QST environment, e.g.:
    OPENBLAS_NUM_THREADS=1 python continuous_step_sweep.py --output validation/continuous_steps

Reads assignments without executing the optimization launcher. Tests its 50 seed
slots at every configured gate time, plus two maximum-amplitude stress controls.
Controls are held fixed as h changes; this is not a repeated optimization sweep.
The batched CFME implementation uses exactly the production floor/remainder grid
and is verified against ML's actual Hamiltonian and local cfme.py before use.
Independent DOP853 reference integrations stop at every pulse boundary.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
from unittest.mock import patch

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import torch

import ML
from cfme import generate_cfme_unitary
from helperFuncs import gateGen, genCouplMat, genDrive, genTwoQuditBasis


def read_config(path):
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith("set -"):
            break
        words = shlex.split(line, comments=True)
        if len(words) == 1 and "=" in words[0]:
            key, value = words[0].split("=", 1)
            values[key] = value
    for key in ("level", "segmentCount", "points", "randomSeedCount", "iterationCount"):
        values[key] = int(values[key])
    for key in ("g", "maxDriveStrength", "anharmonicity", "staggering", "h", "minTime", "maxTime"):
        values[key] = float(values[key])
    if not (values["level"] == 3 and values["crossTalk"] == "True"
            and values["leakage"] == "True" and values["contPulse"] == "True"
            and values["ode"] == "CFME4" and values["maxDriveStrength"] > 0
            and values["couplingType"] != "ContH"):
        raise ValueError("This sweep supports continuous qutrit CTL, bounded controls, static coupling, CFME4.")
    if (not all(np.isfinite(value) for value in values.values() if isinstance(value, (int, float)))
            or values["minTime"] <= 0 or values["maxTime"] < values["minTime"]
            or min(values["segmentCount"], values["points"], values["randomSeedCount"]) < 1):
        raise ValueError("Use finite parameters, positive counts/times, and maxTime >= minTime.")
    return values


class CTLModel:
    def __init__(self, config, weights):
        self.c = config
        self.weights = weights
        self.dim = 16
        self.M = config["segmentCount"]
        self.H0 = config["g"] * genCouplMat(config["couplingType"], 4)
        anharm = np.diag([0, 0, config["anharmonicity"], 3 * config["anharmonicity"]])
        self.static = self.H0 + np.kron(anharm, np.eye(4)) + np.kron(np.eye(4), anharm)
        a = np.diag(np.sqrt(np.arange(1, 4)), 1)
        self.a1, self.a2 = np.kron(a, np.eye(4)), np.kron(np.eye(4), a)
        # Preserve cp()'s actual scalar dtype/convention as well as coefficient order.
        scale = torch.tensor(config["maxDriveStrength"] / np.sqrt(2)).item()
        self.tones = scale * (np.cos(weights[..., :4]) + 1j * np.cos(weights[..., 4:]))
        self.target = gateGen(config["gateType"], 4, 3)
        self.base_time = {"iSWAP": np.pi / 2, "SWAP": 3 * np.pi / 4,
                          "CZ": np.pi / 4, "CNOT": np.pi / 4}[config["gateType"]]

    def H(self, times, segments, T):
        times = np.atleast_1d(times)
        segments = np.broadcast_to(segments, times.shape)
        tones = self.tones[:, segments, :].transpose(1, 0, 2)
        delta, stag = self.c["anharmonicity"], self.c["staggering"]
        phase1 = np.exp(1j * times[:, None] * np.array([0, delta, stag, stag + delta]))
        phase2 = np.exp(1j * times[:, None] * np.array([-stag, delta - stag, 0, delta]))
        shape = np.sin(np.pi * times * self.M / T) ** 2
        d1 = shape[:, None] * np.sum(tones * phase1[:, None, :], axis=-1)
        d2 = shape[:, None] * np.sum(tones * phase2[:, None, :], axis=-1)
        return (self.static + d1[..., None, None] * self.a1
                + d1.conj()[..., None, None] * self.a1.T
                + d2[..., None, None] * self.a2
                + d2.conj()[..., None, None] * self.a2.T)

    def fidelity(self, U):
        tr = np.einsum("ij,...ij->...", self.target.conj(), U)
        return (np.abs(tr) ** 2 + self.dim) / (self.dim * (self.dim + 1))


def production_grid(M, T, h):
    starts, sizes, segments = [], [], []
    duration = T / M
    # Match cfme_with_final_time, including floating-point floor and remainder.
    n = int(duration // h)
    remainder = h * (duration / h - n)
    for m in range(M):
        for j in range(n):
            starts.append(m / M * T + j * h)
            sizes.append(h)
            segments.append(m)
        starts.append(m / M * T + n * h)
        sizes.append(remainder)
        segments.append(m)
    return np.array(starts), np.array(sizes), np.array(segments)


def cfme_batched(model, T, h):
    starts, sizes, segments = production_grid(model.M, T, h)
    U = torch.eye(model.dim, dtype=torch.cdouble).repeat(len(model.weights), 1, 1)
    c1, c2 = (3 - np.sqrt(3)) / 6, (3 + np.sqrt(3)) / 6
    cm, cp = (3 - 2 * np.sqrt(3)) / 12, (3 + 2 * np.sqrt(3)) / 12
    # Bound transient memory while evaluating exponentials in batches.
    chunk = max(1, 1024 // len(model.weights))
    with torch.no_grad():
        for offset in range(0, len(starts), chunk):
            sl = slice(offset, offset + chunk)
            dt = sizes[sl]
            H1 = model.H(starts[sl] + c1 * dt, segments[sl], T)
            H2 = model.H(starts[sl] + c2 * dt, segments[sl], T)
            late = torch.from_numpy(-1j * dt[:, None, None, None] * (cm * H1 + cp * H2))
            early = torch.from_numpy(-1j * dt[:, None, None, None] * (cp * H1 + cm * H2))
            steps = torch.matrix_exp(late) @ torch.matrix_exp(early)
            for step in steps:
                U = step @ U
    return U.numpy()


def reference(model, T, tolerance):
    U = np.broadcast_to(np.eye(model.dim, dtype=complex),
                        (len(model.weights), model.dim, model.dim)).copy()
    evaluations = 0
    for m in range(model.M):
        def rhs(t, y):
            return (-1j * model.H(t, m, T)[0] @ y.reshape(U.shape)).ravel()
        solution = solve_ivp(rhs, (m / model.M * T, (m + 1) / model.M * T),
                             U.ravel(), method="DOP853", rtol=tolerance, atol=tolerance,
                             max_step=T / model.M / 4)
        if not solution.success:
            raise RuntimeError(solution.message)
        U = solution.y[:, -1].reshape(U.shape)
        evaluations += solution.nfev
    return U, evaluations


def verify_production(model):
    """Check actual ML closure, full propagation, and its operator-basis fidelity."""
    T = .1 * model.base_time
    checks = []
    drives = [genDrive(4, l, axis) for l in (1, 2) for axis in ("x", "y")]
    leakage = [genDrive(4, 3, axis) for axis in ("x", "y")]
    anharm = np.diag([0, 0, model.c["anharmonicity"], 2 * model.c["anharmonicity"]])
    class Captured(Exception):
        pass
    def capture(A, t0, duration, h, n, count):
        m = len(checks)
        err = 0.0
        for fraction in (0, .2113, .5, .7887, 1):
            t = t0 + fraction * duration
            expected = -1j * model.H(t, m, T)[0, 0]
            err = max(err, np.max(np.abs(A(t).detach().numpy() - expected)))
        checks.append(float(err))
        if len(checks) == model.M:
            raise Captured
        return torch.eye(model.dim, dtype=torch.cdouble)
    with patch.object(ML.torch, "rand", return_value=torch.tensor(model.weights[0] / (2 * np.pi))), \
            patch.object(ML, "generate_cfme_unitary", side_effect=capture):
        try:
            ML.fidelity_ml(model.M, model.target, T, 1, 1, model.H0, [drives, leakage, anharm],
                           model.c["maxDriveStrength"], True, "False", "True", .01, .5,
                           model.c["anharmonicity"], model.c["staggering"], "CFME4", "True",
                           "SGD", "False", False)
        except Captured:
            pass
    if len(checks) != model.M or max(checks) > 2e-12:
        raise AssertionError(f"Hamiltonian verification failed: {checks}")
    errors = {}
    for h in (.01, .001):
        U = torch.eye(model.dim, dtype=torch.cdouble)
        with torch.no_grad():
            for m in range(model.M):
                A = lambda t: torch.from_numpy(-1j * model.H(t, m, T)[0, 0])
                U = generate_cfme_unitary(A, m / model.M * T, T / model.M, h, 4, 2) @ U
        expected = cfme_batched(CTLModel(model.c, model.weights[:1]), T, h)[0]
        errors[str(h)] = float(np.linalg.norm(U.numpy() - expected) / np.sqrt(model.dim))
        if errors[str(h)] > 2e-12:
            raise AssertionError(f"Production CFME differs: {errors}")
    target = torch.tensor(model.target, dtype=torch.cdouble)
    fid_sum = 0j
    for B in genTwoQuditBasis(4, 4, torch.cdouble):
        fid_sum += torch.trace((target @ B.mH @ target.mH) @ (U @ B @ U.mH)).item()
    fid_basis = abs((fid_sum + model.dim**2) / (model.dim**2 * (model.dim + 1)))
    fidelity_error = float(abs(fid_basis - model.fidelity(U.numpy())))
    if fidelity_error > 2e-12:
        raise AssertionError(f"Fidelity formula differs: {fidelity_error}")
    return {"max_hamiltonian_entry_difference": max(checks),
            "production_propagator_differences": errors,
            "operator_basis_fidelity_difference": fidelity_error}


def make_weights(config, seed_date):
    weights, names, seeds = [], [], []
    for slot in range(config["randomSeedCount"]):
        # ControlFlow hashes date + slot (mod 1000); preserve even duplicate seeds.
        seed = int(hashlib.sha256((seed_date + str(slot)).encode()).hexdigest(), 16) % 1000
        torch.manual_seed(seed)
        weights.append((torch.rand(config["segmentCount"], 8, dtype=torch.double) * 2 * np.pi).numpy())
        names.append(f"random_slot_{slot:02d}_seed_{seed}")
        seeds.append(seed)
    weights.append(np.zeros((config["segmentCount"], 8)))
    names.append("all_tones_maximum")
    weights.append(np.repeat((np.arange(config["segmentCount"]) % 2 * np.pi)[:, None], 8, axis=1))
    names.append("alternating_maximum")
    return np.array(weights), names, seeds


def provenance():
    sources = [Path(__file__), Path(ML.__file__),
               Path(sys.modules["helperFuncs"].__file__), Path(sys.modules["cfme"].__file__)]
    return {"versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                         "torch": torch.__version__, "python": sys.version},
            "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}


def plot_results(rows, output, fidelity_tolerance=1e-8, unitary_tolerance=1e-6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = sorted({r["h"] for r in rows})
    times = sorted({r["normalized_time"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, metric, label, tolerance in zip(
            axes, ("absolute_fidelity_error", "normalized_propagator_error"),
            ("Absolute fidelity error", r"Propagator error $\|U-U_{\rm ref}\|_F/4$"),
            (fidelity_tolerance, unitary_tolerance)):
        for t in times:
            worst = [max(r[metric] for r in rows if r["h"] == h and r["normalized_time"] == t) for h in steps]
            ax.loglog(steps, worst, color="#9baec4", alpha=.55, linewidth=1)
        worst = [max(r[metric] for r in rows if r["h"] == h) for h in steps]
        ax.loglog(steps, worst, "o-", color="#145da0", linewidth=2, markersize=5, label="Worst over all controls and times")
        ax.axhline(tolerance, color="#a33a2b", linestyle="--", label="Acceptance tolerance")
        ax.set(xlabel="Integration step h (in code time units)", ylabel=label)
        ax.grid(True, which="both", alpha=.2)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Continuous qutrit CTL: {len({r['case'] for r in rows})} controls, {len(times)} gate times", fontsize=13)
    fig.savefig(output / "convergence.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("QOC.bash"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-date", default="20261012", help="YYYYMMDD used in ControlFlow's seed hash")
    parser.add_argument("--steps", nargs="+", type=float,
                        default=[.01, .005, .0025, .001, .0005, .00025, .000125])
    parser.add_argument("--fidelity-tolerance", type=float, default=1e-8)
    parser.add_argument("--unitary-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if any(not np.isfinite(x) or x <= 0 for x in args.steps + [args.fidelity_tolerance, args.unitary_tolerance]):
        parser.error("Step sizes and error tolerances must be finite and positive.")
    if len(set(args.steps)) != len(args.steps):
        parser.error("Step sizes must be distinct.")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    config = read_config(args.config)
    weights, names, seeds = make_weights(config, args.seed_date)
    model = CTLModel(config, weights)
    np.savez_compressed(args.output / "controls.npz", weights=weights, names=names)
    (args.output / "QOC.input.bash").write_text(args.config.read_text())
    verification = verify_production(model)
    print("Production verification:", verification, flush=True)
    manifest = {"configuration": config, "seed_date": args.seed_date, "seeds": seeds,
                "cases": names, "steps": args.steps, "production_verification": verification,
                "reference_method": "DOP853, every segment separately, rtol=atol=1e-12",
                "reference_check": "repeat all cases/times at rtol=atol=1e-13",
                "fidelity_tolerance": args.fidelity_tolerance,
                "normalized_propagator_tolerance": args.unitary_tolerance,
                "scope": "Frozen initial controls and amplitude-bound stress controls; no reoptimization.",
                **provenance()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    ref_checks = []
    started = time.perf_counter()
    fieldnames = ["normalized_time", "T", "pulse_duration", "case", "h", "steps_per_pulse",
                  "fidelity", "reference_fidelity", "absolute_fidelity_error",
                  "normalized_propagator_error", "normalized_unitarity_error"]
    with (args.output / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for normalized_time in np.linspace(config["minTime"], config["maxTime"], config["points"]):
            T = normalized_time * model.base_time
            U_ref, nfev = reference(model, T, 1e-12)
            U_tight, nfev_tight = reference(model, T, 1e-13)
            ref_difference = np.linalg.norm(U_ref - U_tight, axis=(-2, -1)) / np.sqrt(model.dim)
            ref_fidelity_difference = np.abs(model.fidelity(U_ref) - model.fidelity(U_tight))
            ref_checks.append({"normalized_time": float(normalized_time),
                               "max_normalized_propagator_difference": float(ref_difference.max()),
                               "max_fidelity_difference": float(ref_fidelity_difference.max()),
                               "nfev": nfev, "nfev_tight": nfev_tight})
            if ref_difference.max() > 1e-8 or ref_fidelity_difference.max() > 1e-10:
                raise RuntimeError(f"Reference failed convergence check: {ref_checks[-1]}")
            reference_fidelity = model.fidelity(U_tight)
            np.savez_compressed(args.output / f"reference_t{normalized_time:.4f}.npz", U=U_tight)
            print(f"T/Tmin={normalized_time:.1f}: reference verified, max dU={ref_difference.max():.3g}", flush=True)
            for h in args.steps:
                tic = time.perf_counter()
                U = cfme_batched(model, T, h)
                fidelities = model.fidelity(U)
                fid_errors = np.abs(fidelities - reference_fidelity)
                U_errors = np.linalg.norm(U - U_tight, axis=(-2, -1)) / np.sqrt(model.dim)
                unitarity = np.linalg.norm(U.conj().transpose(0, 2, 1) @ U - np.eye(model.dim), axis=(-2, -1)) / np.sqrt(model.dim)
                for i, name in enumerate(names):
                    row = dict(zip(fieldnames, [float(normalized_time), float(T), float(T / model.M),
                               name, h, int(np.count_nonzero(production_grid(1, T / model.M, h)[1])),
                               float(fidelities[i]), float(reference_fidelity[i]), float(fid_errors[i]),
                               float(U_errors[i]), float(unitarity[i])]))
                    rows.append(row)
                    writer.writerow(row)
                f.flush()
                print(f"  h={h:g}: max |dF|={fid_errors.max():.3g}, max dU={U_errors.max():.3g} ({time.perf_counter()-tic:.1f}s)", flush=True)
            (args.output / "reference_checks.json").write_text(json.dumps(ref_checks, indent=2))
    summary = []
    for h in args.steps:
        subset = [r for r in rows if r["h"] == h]
        max_fid = max(subset, key=lambda r: r["absolute_fidelity_error"])
        max_U = max(subset, key=lambda r: r["normalized_propagator_error"])
        summary.append({"h": h, "max_absolute_fidelity_error": max_fid["absolute_fidelity_error"],
                        "max_normalized_propagator_error": max_U["normalized_propagator_error"],
                        "max_normalized_unitarity_error": max(r["normalized_unitarity_error"] for r in subset),
                        "worst_fidelity_case": max_fid, "worst_propagator_case": max_U,
                        "passes": max_fid["absolute_fidelity_error"] <= args.fidelity_tolerance
                                  and max_U["normalized_propagator_error"] <= args.unitary_tolerance})
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["summary"] = summary
    passing = [r["h"] for r in summary if r["passes"]]
    manifest["largest_tested_passing_h"] = max(passing) if passing else None
    manifest["reference_convergence"] = ref_checks
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    plot_results(rows, args.output, args.fidelity_tolerance, args.unitary_tolerance)
    print(json.dumps(summary, indent=2), flush=True)
    print("Largest tested passing h:", manifest["largest_tested_passing_h"], flush=True)


if __name__ == "__main__":
    main()
