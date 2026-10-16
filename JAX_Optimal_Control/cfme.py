"""JAX commutator-free Magnus propagation, preserving the original ordering.

The original CFME implementation always includes a final remainder step, even
when its duration is zero.  Keep that grid here so matched controls can be
compared directly between the two backends.
"""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import expm

__version__ = "jax-v0.1.0"


def time_grid(t0, T, h):
    if h <= 0 or T < 0:
        raise ValueError("h must be positive and T nonnegative")
    count = int(T // h)
    sizes = np.concatenate((np.full(count, h), [h * (T / h - count)]))
    return t0 + h * np.arange(count + 1), sizes


def special_inversion_matrix(s):
    if s == 2:
        values = ((1, 0), (0, 12))
    elif s == 3:
        values = ((9 / 4, 0, 15), (0, 12, 0), (-15, 0, 180))
    else:
        raise ValueError("Only s=2 or s=3 is implemented")
    return jnp.asarray(values, dtype=jnp.complex128)


def legendre_gauss_quad_matrix(s, r):
    nodes, weights = np.polynomial.legendre.leggauss(r)
    nodes = (nodes + 1) / 2
    return jnp.asarray([weights * (nodes - .5) ** i / 2 for i in range(s)],
                       dtype=jnp.complex128)


def commutator_free_magnus_operators(A, t, h, s, r, i):
    nodes = (np.polynomial.legendre.leggauss(r)[0] + 1) / 2
    coefficients = special_inversion_matrix(s) @ legendre_gauss_quad_matrix(s, r)
    samples = jnp.stack([A(t + node * h) for node in nodes])
    return h * jnp.einsum("q,qij->ij", coefficients[i - 1], samples)


def fourth_order_cfme(A, t0, h, n=4, m=2):
    b1 = commutator_free_magnus_operators(A, t0, h, 2, 2, 1)
    b2 = commutator_free_magnus_operators(A, t0, h, 2, 2, 2)
    if m == 2:
        generators = (.5 * b1 + b2 / 6, .5 * b1 - b2 / 6)
    elif m == 3:
        generators = (b2 / 12, b1, -b2 / 12)
    else:
        raise NotImplementedError("Fourth-order CFME supports m=2 or m=3")
    return jnp.stack([expm(generator) for generator in generators])


def sixth_order_cfme(A, t0, h, n=6, m=5):
    b1, b2, b3 = [commutator_free_magnus_operators(A, t0, h, 3, 4, i)
                  for i in (1, 2, 3)]
    if m == 5:
        a, b, c = .2, .08734395950888931101, .03734395950888931101
        d, e, f = .34815492558797391479, .053438272547684150, .00584269157837031012
        g, k = 1 - 2 * (a + d), 1 / 12 - 2 * (c + f)
        coefficients = ((a, b, c), (d, e, f), (g, 0, k), (d, -e, f), (a, -b, c))
    elif m == 6:
        a, b, c = .208, .09023186422416794596, .03823186422416794596
        d, e, f = .312, .04467385661651479788, .00439421553992544024
        g, k, l = .5 - a - d, .01407960659498524468, 1 / 24 - c - f
        coefficients = ((a, b, c), (d, e, f), (g, k, l),
                        (g, -k, l), (d, -e, f), (a, -b, c))
    else:
        raise NotImplementedError("Sixth-order CFME supports m=5 or m=6")
    return jnp.stack([expm(a * b1 + b * b2 + c * b3) for a, b, c in coefficients])


def cfme_timestep(A, t0, h, n, m):
    if n == 4:
        return fourth_order_cfme(A, t0, h, n, m)
    if n == 6:
        return sixth_order_cfme(A, t0, h, n, m)
    raise ValueError("CFME order must be 4 or 6")


def cfme(A, t0, h, N, n, m):
    return [cfme_timestep(A, t0 + h * i, h, n, m) for i in range(N)]


def cfme_with_final_time(A, t0, T, h, n, m):
    starts, sizes = time_grid(t0, T, h)
    return [cfme_timestep(A, t, dt, n, m) for t, dt in zip(starts, sizes)]


def generate_unitary_from_operator_stack(operators):
    operators = jnp.asarray(operators, dtype=jnp.complex128)
    dim = operators.shape[-1]
    def step(result, timestep):
        def multiply(value, operator):
            return operator @ value, None
        result, _ = jax.lax.scan(multiply, result, timestep[::-1])
        return result, None
    return jax.lax.scan(step, jnp.eye(dim, dtype=jnp.complex128), operators)[0]


def generate_cfme_unitary(A, t0, T, h, n, m):
    starts, sizes = time_grid(t0, T, h)
    # scan avoids unrolling the time grid when this helper is jitted. A must
    # accept JAX scalar times; the training API precomputes time-only data.
    dim = A(float(t0)).shape[-1]
    def step(unitary, time_and_step):
        time, dt = time_and_step
        ops = cfme_timestep(A, time, dt, n, m)
        def multiply(value, operator):
            return operator @ value, None
        unitary, _ = jax.lax.scan(multiply, unitary, ops[::-1])
        return unitary, None
    return jax.lax.scan(step, jnp.eye(dim, dtype=jnp.complex128),
                        (jnp.asarray(starts), jnp.asarray(sizes)))[0]


def apply_operator_stack(state, operators):
    def step(value, timestep):
        def multiply(current, operator):
            return operator @ current, None
        value, _ = jax.lax.scan(multiply, value, timestep[::-1])
        return value / jnp.linalg.norm(value), None
    return jax.lax.scan(step, jnp.asarray(state), jnp.asarray(operators))[0]


def dag(operator):
    return jnp.conj(operator.T)


def make_unitary(operator):
    # The corresponding PyTorch function is also an identity operation.
    return operator
