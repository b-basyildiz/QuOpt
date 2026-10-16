"""Check the pinned environment and run real QuOpt calculations on a compute node.

Default target: Caltech CPython 3.11/Linux CPU. --expected-python 3.10 is used
only to validate this runner in the existing local reference environment.
Production Data/ is untouched: pilot outputs go to a temporary directory.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parent
REFERENCE_FIDELITY = 0.06496649980545044
REFERENCE_TOLERANCE = 1e-6


def check_environment(expected_python):
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual != expected_python:
        raise RuntimeError(f"Expected Python {expected_python}; got {sys.version}.")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the QuOpt virtual environment, not the base Python.")
    target = os.environ.get("QUTRIT_ENV")
    if target and Path(sys.prefix).resolve() != Path(target).resolve():
        raise RuntimeError(f"Running in {sys.prefix}, but QUTRIT_ENV={target}.")
    expected = {}
    for line in (ROOT / "requirements-caltech.txt").read_text().splitlines():
        line = line.partition("#")[0].strip()
        if line:
            name, version = line.split("==")
            expected[name] = version
    expected["torch"] = "1.13.1+cpu" if sys.platform == "linux" else "1.13.1"
    versions = {}
    for name, version in expected.items():
        actual_version = importlib.metadata.version(name)
        if actual_version != version:
            raise RuntimeError(f"{name}: expected {version}, got {actual_version}.")
        importlib.import_module(name)
        versions[name] = actual_version
        print(f"{name} == {actual_version}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    return versions


def production_pilot(directory, normalized_time, iterations, repeat):
    import numpy as np

    args = ["iSWAP", "3", "SpeedUp", "40", "1", "14", "True", "17",
            "CFME4", "0.0005", "0.5", "True", "True", "False", "40",
            str(normalized_time), str(normalized_time), "-1", str(iterations),
            "SGD", "0", "0", "-1"]
    started = time.perf_counter()
    # points=-1 selects rseed=1, so this check does not change with the date.
    subprocess.run([sys.executable, "-u", str(ROOT / "ControlFlow.py"), *args],
                   cwd=directory, check=True)
    data = Path(directory) / "Data/CTL/SpeedUp"
    fidelity_paths = list((data / "Fidelities").glob("*.csv"))
    weight_paths = list((data / "Weights").rglob("Weights_*_best.csv"))
    if len(fidelity_paths) != 1 or len(weight_paths) != 1:
        raise RuntimeError("Pilot did not produce exactly one fidelity file and one weight file.")
    result = np.loadtxt(fidelity_paths[0], delimiter=",", ndmin=2)
    weights = np.loadtxt(weight_paths[0], delimiter=",", skiprows=1)
    if result.shape != (1, 2) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid fidelity output: {result}.")
    if weights.shape != (40, 8) or not np.isfinite(weights).all():
        raise RuntimeError("Expected finite 40 x 8 optimized weights.")
    fidelity = float(result[0, 0])
    if abs(float(result[0, 1]) - normalized_time) > 1e-12 or not 0 <= fidelity <= 1 + 1e-8:
        raise RuntimeError(f"Unexpected time/fidelity output: {result}.")
    if normalized_time == .1 and abs(fidelity - REFERENCE_FIDELITY) > REFERENCE_TOLERANCE:
        raise RuntimeError(f"F={fidelity:.12g} differs from the reference "
                           f"{REFERENCE_FIDELITY:.12g} by more than {REFERENCE_TOLERANCE}.")
    record = {"normalized_time": normalized_time, "iterations": iterations,
              "repeat": repeat, "fidelity": fidelity,
              "elapsed_seconds": time.perf_counter() - started,
              "weights_shape": list(weights.shape)}
    print("Production pilot:", json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-python", default="3.11")
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    identifier = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
    destination = ROOT / "smoke-results" / f"QuOpt-{identifier}.json"
    destination.parent.mkdir(exist_ok=True)
    report = {"status": "RUNNING", "python": sys.version,
              "executable": sys.executable, "platform": platform.platform(),
              "reference_fidelity": REFERENCE_FIDELITY,
              "reference_absolute_tolerance": REFERENCE_TOLERANCE,
              "pilots": [],
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("ControlFlow.py", "ML.py", "helperFuncs.py", "cfme.py")}}
    try:
        print("Python:", sys.version, "\nExecutable:", sys.executable, flush=True)
        report["packages"] = check_environment(args.expected_python)
        subprocess.run([sys.executable, "-m", "unittest", "test_bugfixes", "test_continuous_pulses", "-v"],
                       cwd=ROOT, check=True)
        report["regression_tests"] = "PASSED"
        with tempfile.TemporaryDirectory(prefix="QuOpt-smoke-") as temporary:
            short = Path(temporary) / "short"
            long = Path(temporary) / "long"
            short.mkdir()
            long.mkdir()
            # Repeat in the same directory to check the existing-CSV code path.
            for repeat in (1, 2):
                report["pilots"].append(production_pilot(short, .1, 3, repeat))
            report["pilots"].append(production_pilot(long, 1., 2, 1))
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                text=True, capture_output=True, check=True).stdout
        (destination.parent / f"requirements-tested-{identifier}.txt").write_text(frozen)
        report["status"] = "PASSED"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = str(error)
        traceback.print_exc()
    finally:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print("Report:", destination, flush=True)
    if report["status"] != "PASSED":
        return 1
    print("QUOPT SMOKE TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
