"""Native JAX control optimization with the original fidelity_ml interface.

All propagation and differentiation use float64/complex128.  Time grids,
quadrature constants, and pulse phases are built once on the host; JIT-compiled
scans then reuse them for every parameter update. PyTorch is not a dependency.
"""
from dataclasses import dataclass
import math
import time
import warnings

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import expm

try:
    from .cfme import time_grid
except ImportError:
    from cfme import time_grid


def _flag(value):
    return value is True or value == "True"


@dataclass
class Problem:
    """Compiled functions for a fixed physical model and integration grid."""
    forward: object
    loss: object
    evaluate: object
    value_and_grad: object
    step: object
    adam_step: object
    weight_shape: tuple
    metadata: dict


def build_problem(M, input_gate, tmin, H0, drives, maxDriveStrength,
                  lbool, minLeak, crossTalk, h, alpha, anharmVal, stag,
                  ode, ContPulse):
    """Build a reusable problem, using the same arguments as fidelity_ml.

    ``evaluate(weights)`` returns ``(cost, (infidelity, unitary, leakage_cost))``.
    ``step(weights, momentum, lr)`` performs a complete Nesterov SGD update and
    returns new weights, new momentum, cost, the same auxiliary tuple, gradient.
    Initialize momentum with zeros. ``value_and_grad`` differentiates cost only.
    """
    M = int(M)
    if M < 1 or tmin <= 0 or h <= 0:
        raise ValueError("M, total gate time, and h must be positive")
    leakage, cross, continuous = _flag(lbool), _flag(crossTalk), _flag(ContPulse)
    minimize_leakage = _flag(minLeak)
    if leakage:
        drives, leakage_drives, anharmonicity = drives
        leakage_drives = np.asarray(leakage_drives, dtype=np.complex128)
        anharmonicity = np.asarray(anharmonicity, dtype=np.complex128)
    drives = np.asarray(drives, dtype=np.complex128)
    if drives.ndim != 3 or drives.shape[1] != drives.shape[2]:
        raise ValueError("drives must contain square matrices of equal size")
    level = drives.shape[-1]
    dim = level ** 2
    target = jnp.asarray(input_gate, dtype=jnp.complex128)
    if target.shape != (dim, dim):
        raise ValueError("Incorrect target gate size")
    ctl = leakage and cross and level >= 4
    nparams = 8 if ctl else 2 * len(drives)
    timed_coupling = callable(H0)
    static = not (ctl or continuous or cross or timed_coupling)
    if not static and ode not in ("CFME4", "RK2", "SRK2"):
        raise ValueError("ode must be CFME4, RK2, or SRK2 for time-dependent models")
    method = "static" if static else ode
    identity = np.eye(level, dtype=np.complex128)
    identity_full = jnp.eye(dim, dtype=jnp.complex128)

    # Each segment contains the exact production grid, including CFME's final
    # remainder. RK methods rescale to ceil((tf-t0)/h) steps per segment.
    times_list, sizes_list = [], []
    nodes = (np.polynomial.legendre.leggauss(2)[0] + 1) / 2
    for segment in range(M):
        start = segment / M * tmin
        end = (segment + 1) / M * tmin
        if method == "CFME4":
            starts, sizes = time_grid(start, tmin / M, h)
            times = starts[:, None] + sizes[:, None] * nodes
        elif method in ("RK2", "SRK2"):
            count = math.ceil((end - start) / h)
            dt = (end - start) / count
            sizes = np.full(count, dt)
            # Match the original repeatedly incremented time variable.
            starts = np.empty(count)
            current = start
            for index in range(count):
                starts[index] = current
                current += dt
            offsets = (0, .5) if method == "RK2" else (.25, .75)
            times = starts[:, None] + sizes[:, None] * np.asarray(offsets)
        else:
            sizes = np.asarray([tmin / M])
            times = np.asarray([[0.]])
        times_list.append(times)
        sizes_list.append(sizes)
    nsteps = max(len(values) for values in sizes_list)
    times = np.stack([np.pad(values, ((0, nsteps-len(values)), (0, 0)), mode="edge")
                      for values in times_list])
    sizes = jnp.asarray(np.stack([np.pad(values, (0, nsteps-len(values)))
                                 for values in sizes_list]))
    envelope = (np.sin(np.pi * times * M / tmin) ** 2 if continuous
                else np.ones_like(times))
    envelope = jnp.asarray(envelope)
    if timed_coupling:
        coupling = np.stack([np.asarray(H0(float(t)), dtype=np.complex128)
                             for t in times.flat]).reshape(times.shape + (dim, dim))
    else:
        coupling = np.asarray(H0, dtype=np.complex128)
    if coupling.shape[-2:] != (dim, dim):
        raise ValueError("Incorrect coupling Hamiltonian size")
    coupling = jnp.asarray(coupling)

    if ctl:
        diagonal = np.diag(np.arange(level) * (np.arange(level)-1) * anharmVal / 2)
        drift = jnp.asarray(np.kron(diagonal, identity) + np.kron(identity, diagonal))
        ladder = np.diag(np.sqrt(np.arange(1, level)), 1)
        ladder1 = jnp.asarray(np.kron(ladder, identity))
        ladder2 = jnp.asarray(np.kron(identity, ladder))
        phases1 = jnp.asarray(np.exp(1j * times[..., None] *
                                     np.asarray([0, anharmVal, stag, stag+anharmVal])))
        phases2 = jnp.asarray(np.exp(1j * times[..., None] *
                                     np.asarray([-stag, anharmVal-stag, 0, anharmVal])))
        # torch.tensor(maxDriveStrength/np.sqrt(2)) receives a NumPy float64.
        cap = float(maxDriveStrength / np.sqrt(2))

        def hamiltonians(weights, segment):
            amplitudes = cap * (jnp.cos(weights[:4]) + 1j*jnp.cos(weights[4:8]))
            d1 = envelope[segment] * jnp.sum(phases1[segment] * amplitudes, axis=-1)
            d2 = envelope[segment] * jnp.sum(phases2[segment] * amplitudes, axis=-1)
            controls = (d1[..., None, None] * ladder1 + d1.conj()[..., None, None] * ladder1.T +
                        d2[..., None, None] * ladder2 + d2.conj()[..., None, None] * ladder2.T)
            return (coupling[segment] if timed_coupling else coupling) + drift + controls
    else:
        operators = np.stack([[np.kron(drive, identity), np.kron(identity, drive)]
                              for drive in drives])
        if leakage:
            for index in range(len(drives)):
                if index >= level - 2:
                    extra = leakage_drives[index % len(leakage_drives)]
                    operators[index, 0] += np.kron(extra, identity)
                    operators[index, 1] += np.kron(identity, extra)
            drift = np.kron(identity, anharmonicity) + np.kron(anharmonicity, identity)
        else:
            drift = np.zeros((dim, dim), dtype=np.complex128)
        operators = jnp.asarray(operators)
        drift = jnp.asarray(drift)
        if cross:
            cross_ops = jnp.asarray(np.stack([[np.kron(drive, identity), np.kron(identity, drive)]
                                              for drive in drives]))
            phases = jnp.asarray(np.exp(1j * times[..., None] * np.asarray([-stag, stag])))
            if leakage:
                if len(leakage_drives) < len(drives):
                    raise ValueError("Original non-CTL crosstalk model requires one leakage drive per drive")
                cross_leak_ops = jnp.asarray(np.stack([[np.kron(drive, identity), np.kron(identity, drive)]
                                                       for drive in leakage_drives[:len(drives)]]))
                leak_phases = jnp.asarray(np.exp(1j * times[..., None] *
                                                np.asarray([-stag-anharmVal, stag+anharmVal])))

        def hamiltonians(weights, segment):
            amplitudes = weights.reshape((-1, 2))
            if maxDriveStrength != -1:
                amplitudes = maxDriveStrength * jnp.cos(amplitudes)
            control = envelope[segment, ..., None, None] * jnp.einsum("pq,pqij->ij", amplitudes, operators)
            if cross:
                ct = jnp.einsum("pq,stq,pqij->stij", amplitudes[:, ::-1], phases[segment], cross_ops)
                if leakage:
                    ct = ct + jnp.einsum("pq,stq,pqij->stij", amplitudes[:, ::-1],
                                         leak_phases[segment], cross_leak_ops)
                ct = envelope[segment, ..., None, None] * ct
                ct = ct + jnp.swapaxes(ct.conj(), -1, -2)
                if not continuous:
                    # Production's torch.tensor(H1ct(t)) intentionally detaches
                    # this term. Preserve its gradient behavior for parity.
                    ct = jax.lax.stop_gradient(ct)
                control = control + ct
            return (coupling[segment] if timed_coupling else coupling) + drift + control

    # The original qtt state mask is all states with either qudit in its top
    # level. Its qb mask selects |00>, |01>, |10>, |11>.
    high_states = jnp.asarray([i for i in range(dim) if i // level == level-1 or i % level == level-1])
    low_states = jnp.asarray([0, 1, level, level+1])
    def occupancy(unitary):
        return jnp.sum(jnp.abs(unitary[high_states[:, None], low_states[None, :]]) ** 2) / 4

    # The two CFME moments below follow the original b1/b2 construction;
    # op2 is applied first, then op1, in chronological timestep order.
    node_weights = np.polynomial.legendre.leggauss(2)[1] / 2
    b1_coeff = jnp.asarray(node_weights)
    b2_coeff = jnp.asarray(12 * node_weights * (nodes - .5))
    def integrate(unitary, sample):
        hamiltonian, dt = sample
        if method == "CFME4":
            b1 = -1j * dt * jnp.einsum("q,qij->ij", b1_coeff, hamiltonian)
            b2 = -1j * dt * jnp.einsum("q,qij->ij", b2_coeff, hamiltonian)
            unitary = expm(.5*b1 - b2/6) @ unitary
            unitary = expm(.5*b1 + b2/6) @ unitary
        elif method == "RK2":
            k1 = -1j * hamiltonian[0] @ unitary
            k2 = -1j * hamiltonian[1] @ (unitary + .5*dt*k1)
            unitary = unitary + dt*k2
        elif method == "SRK2":
            h1, h2 = hamiltonian
            k1 = jnp.linalg.solve(identity_full + 1j*dt*.25*h1, -1j*h1 @ unitary)
            k2 = jnp.linalg.solve(identity_full + 1j*dt*.25*h2,
                                  -1j*h2 @ unitary - 1j*.5*dt*h2 @ k1)
            unitary = unitary + .5*dt*(k1+k2)
        else:
            unitary = expm(-1j*dt*hamiltonian[0]) @ unitary
        return unitary, None

    def propagate(weights):
        def segment_step(unitary, segment):
            matrices = hamiltonians(weights[segment], segment)
            unitary, _ = jax.lax.scan(integrate, unitary, (matrices, sizes[segment]))
            if method == "RK2":
                unitary = unitary / jnp.sqrt(jnp.trace(unitary.conj().T @ unitary) / dim)
            return unitary, occupancy(unitary) if minimize_leakage else jnp.float64(0)
        unitary, occupations = jax.lax.scan(segment_step, identity_full, jnp.arange(M))
        leakage_cost = jnp.float64(0)
        if minimize_leakage:
            occupations = jnp.concatenate((occupancy(identity_full)[None], occupations))
            # torchMax keeps the first maximum when values tie.
            maximum = occupations[jnp.argmax(occupations)]
            leakage_cost = alpha * (jnp.mean(occupations) + maximum)
        return unitary, leakage_cost

    def evaluate(weights):
        unitary, leakage_cost = propagate(weights)
        overlap = jnp.trace(target.conj().T @ unitary)
        # Complete orthogonal operator-basis identity: exactly the original
        # basis sum, including its constant, even for a projected target.
        fidelity = (jnp.abs(overlap) ** 2 + dim) / (dim * (dim+1))
        infidelity = 1 - fidelity
        return infidelity + leakage_cost, (infidelity, unitary, leakage_cost)

    value_grad_aux = jax.value_and_grad(evaluate, has_aux=True)
    @jax.jit
    def step(weights, momentum, learning_rate):
        (cost, aux), gradient = value_grad_aux(weights)
        momentum = .99 * momentum + gradient
        weights = weights - learning_rate * (gradient + .99 * momentum)
        return weights, momentum, cost, aux, gradient

    @jax.jit
    def adam_step(weights, first, second, count, learning_rate):
        (cost, aux), gradient = value_grad_aux(weights)
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient ** 2
        count = count + 1
        corrected_first = first / (1 - .9 ** count)
        corrected_second = second / (1 - .999 ** count)
        weights = weights - learning_rate * corrected_first / (jnp.sqrt(corrected_second) + 1e-8)
        return weights, first, second, count, cost, aux, gradient

    loss = lambda weights: evaluate(weights)[0]
    return Problem(jax.jit(lambda weights: propagate(weights)[0]), jax.jit(loss),
                   jax.jit(evaluate), jax.jit(jax.value_and_grad(loss)), step, adam_step,
                   (M, nparams), {"level": level, "dimension": dim, "method": method,
                                  "steps_per_segment": nsteps, "total_steps": int(sum(map(len, sizes_list))),
                                  "continuous": continuous, "ctl": ctl, "x64": True})


class Scheduler:
    """Preserve the original training loop's scheduler calls.

    The original passes infidelity to every scheduler. CosineAnnealingLR treats
    that value as its epoch, so this mode intentionally preserves that legacy
    behavior rather than using an iteration-based cosine schedule.
    """
    def __init__(self, optimizer):
        if optimizer not in ("SGD", "ADAM", "CosineLR"):
            raise ValueError("optimizer must be SGD, ADAM, or CosineLR")
        self.optimizer = optimizer
        self.lr = .3
        self.best = math.inf
        self.bad_epochs = 0
        self.epoch = 0

    def step(self, infidelity):
        self.epoch += 1
        if self.optimizer == "CosineLR":
            self.epoch = float(infidelity)
            self.lr = .3 * (1 + math.cos(math.pi * self.epoch / 50)) / 2
        else:
            if float(infidelity) < self.best * (1 - 1e-4):
                self.best = float(infidelity)
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            if self.bad_epochs > 20:
                new_lr = max(self.lr * .3, .03)
                if self.lr - new_lr > 1e-8:
                    self.lr = new_lr
                self.bad_epochs = 0
        return self.lr


def make_scheduler(optimizer):
    return Scheduler(optimizer)


def fidelity_ml(M, input_gate, tmin, N_iter, rseed, H0, drives,
                maxDriveStrength, lbool, minLeak, crossTalk, h, alpha,
                anharmVal, stag, ode, ContPulse, optimizer, weightFName,
                wsmBoolF, *, initial_weights=None, return_diagnostics=False):
    """Train and return ``[best_fidelity, weights]`` like the PyTorch pipeline.

    JAX's seeded RNG is deliberately independent of PyTorch's RNG. Supply the
    same ``initial_weights`` (or warm-start CSV) for numerical comparisons.
    With ``return_diagnostics=True``, return ``(legacy_result, diagnostics)``;
    recorded per-iteration fidelities/weights are evaluated before each update.
    The legacy return rounds infidelity to float32 before selecting its minimum.
    """
    if N_iter < 1:
        raise ValueError("N_iter must be at least one")
    problem = build_problem(M, input_gate, tmin, H0, drives, maxDriveStrength,
                            lbool, minLeak, crossTalk, h, alpha, anharmVal,
                            stag, ode, ContPulse)
    scheduler = make_scheduler(optimizer)
    warm_start = weightFName != "False"
    if initial_weights is None and warm_start:
        try:
            initial_weights = np.loadtxt(weightFName, delimiter=",", skiprows=1, ndmin=2)
        except (OSError, ValueError) as error:
            warnings.warn(f"Could not load warm-start weights; initializing a new seed: {error}")
    if initial_weights is None:
        weights = jax.random.uniform(jax.random.PRNGKey(rseed), problem.weight_shape,
                                     dtype=jnp.float64, minval=0., maxval=2*np.pi)
    else:
        initial_weights = np.asarray(initial_weights, dtype=np.float64)
        if initial_weights.shape != problem.weight_shape or not np.isfinite(initial_weights).all():
            raise ValueError(f"initial_weights must be finite with shape {problem.weight_shape}")
        weights = jnp.asarray(initial_weights)
    momentum = jnp.zeros_like(weights)
    second = jnp.zeros_like(weights)
    count = jnp.asarray(0, dtype=jnp.int64)
    best_infidelity = np.float32(np.inf)
    best_weights = weights
    diagnostics = {key: [] for key in ("unitaries", "weights_before", "weights_after",
                    "gradients", "fidelities", "infidelities", "costs", "leakage",
                    "learning_rates_before", "learning_rates_after", "iteration_seconds")}
    diagnostics["metadata"] = problem.metadata
    for _ in range(N_iter):
        previous_weights = weights
        learning_rate = scheduler.lr
        start = time.perf_counter()
        if optimizer == "ADAM":
            weights, momentum, second, count, cost, aux, gradient = problem.adam_step(
                weights, momentum, second, count, jnp.float64(learning_rate))
        else:
            weights, momentum, cost, aux, gradient = problem.step(
                weights, momentum, jnp.float64(learning_rate))
        # A complete training step must be synchronized before recording time.
        weights.block_until_ready()
        infidelity, unitary, leakage_cost = aux
        metric = float(infidelity)
        if not (math.isfinite(metric) and math.isfinite(float(cost))):
            raise FloatingPointError("Non-finite training objective; check the model and step size")
        rounded = np.float32(metric)
        if rounded < best_infidelity:
            best_infidelity, best_weights = rounded, previous_weights
        scheduler.step(metric)
        if return_diagnostics:
            for key, value in (("unitaries", unitary), ("weights_before", previous_weights),
                               ("weights_after", weights), ("gradients", gradient)):
                diagnostics[key].append(np.asarray(value).copy())
            for key, value in (("fidelities", 1-metric), ("infidelities", metric),
                               ("costs", cost), ("leakage", leakage_cost),
                               ("learning_rates_before", learning_rate),
                               ("learning_rates_after", scheduler.lr)):
                diagnostics[key].append(float(value))
            diagnostics["iteration_seconds"].append(time.perf_counter() - start)
    selected = weights if warm_start and not wsmBoolF else best_weights
    result = [1 - float(best_infidelity), np.asarray(selected).copy()]
    if return_diagnostics:
        diagnostics["first_iteration_seconds"] = diagnostics["iteration_seconds"][0]
        diagnostics["warmed_iteration_seconds"] = diagnostics["iteration_seconds"][1:]
        return result, diagnostics
    return result
