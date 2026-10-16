"""Small analytic checks; benchmark_parity.py tests the original implementation."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from ML import build_problem, make_scheduler


class JaxControlTests(unittest.TestCase):
    def problem(self, continuous="True", cap=2., h=.03):
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.H0 = np.diag([.2, -.3, .7, -.4]).astype(complex)
        target = np.eye(4, dtype=complex)
        target[1, 1] = target[2, 2] = 0
        target[1, 2] = target[2, 1] = 1j
        return build_problem(2, target, .17, self.H0, [x, y], cap,
                             False, "False", "False", h, .5, 14., 17.,
                             "CFME4", continuous)

    def test_double_precision_enabled(self):
        self.assertTrue(jax.config.x64_enabled)
        problem = self.problem()
        unitary = problem.forward(jnp.zeros((2, 4), dtype=jnp.float64))
        self.assertEqual(unitary.dtype, jnp.complex128)

    def test_zero_drive_matches_exact_drift_for_both_pulse_types(self):
        for pulse in ("True", "False"):
            with self.subTest(continuous=pulse):
                problem = self.problem(continuous=pulse, cap=0)
                actual = np.asarray(problem.forward(jnp.ones((2, 4))))
                np.testing.assert_allclose(actual, expm(-1j * self.H0 * .17), atol=2e-13, rtol=0)

    def test_gradient_matches_central_difference(self):
        problem = self.problem()
        weights = np.random.default_rng(19).uniform(.2, 5.9, (2, 4))
        value, gradient = problem.value_and_grad(jnp.asarray(weights))
        self.assertTrue(np.isfinite(float(value)))
        delta = 1e-5
        finite_difference = np.zeros_like(weights)
        for index in np.ndindex(weights.shape):
            plus, minus = weights.copy(), weights.copy()
            plus[index] += delta
            minus[index] -= delta
            finite_difference[index] = (float(problem.loss(plus)) - float(problem.loss(minus))) / (2 * delta)
        np.testing.assert_allclose(np.asarray(gradient), finite_difference, atol=2e-9, rtol=2e-6)

    def test_nesterov_updates_match_equations(self):
        problem = self.problem(continuous="False")
        weights = jnp.asarray(np.random.default_rng(23).uniform(.2, 5.9, (2, 4)))
        momentum = jnp.zeros_like(weights)
        for _ in range(3):
            _, gradient = problem.value_and_grad(weights)
            expected_momentum = .99 * momentum + gradient
            expected_weights = weights - .3 * (gradient + .99 * expected_momentum)
            actual = problem.step(weights, momentum, .3)
            np.testing.assert_allclose(actual[0], expected_weights, atol=2e-13, rtol=0)
            np.testing.assert_allclose(actual[1], expected_momentum, atol=2e-13, rtol=0)
            weights, momentum = actual[:2]

    def test_plateau_waits_more_than_patience_and_honors_floor(self):
        scheduler = make_scheduler("SGD")
        learning_rates = [scheduler.step(1.) for _ in range(64)]
        np.testing.assert_allclose(learning_rates[:21], .3, rtol=0, atol=1e-15)
        np.testing.assert_allclose(learning_rates[21:42], .09, rtol=0, atol=1e-15)
        np.testing.assert_allclose(learning_rates[42:], .03, rtol=0, atol=1e-15)

    def test_plateau_uses_relative_improvement_threshold(self):
        scheduler = make_scheduler("SGD")
        scheduler.step(1.)
        for _ in range(20):
            scheduler.step(.99995)  # A decrease smaller than 1e-4 is not an improvement.
        self.assertAlmostEqual(scheduler.step(.99995), .09)
        scheduler = make_scheduler("SGD")
        scheduler.step(1.)
        for _ in range(20):
            scheduler.step(1.)
        self.assertAlmostEqual(scheduler.step(.9998), .3)
        for _ in range(20):
            self.assertAlmostEqual(scheduler.step(.9998), .3)
        self.assertAlmostEqual(scheduler.step(.9998), .09)

    def test_cosine_schedule_matches_closed_form(self):
        scheduler = make_scheduler("CosineLR")
        for epoch in range(1, 56):
            actual = scheduler.step(epoch)
            expected = .3 * (1 + np.cos(np.pi * epoch / 50)) / 2
            self.assertAlmostEqual(actual, expected, places=13)

    def test_cosine_preserves_legacy_infidelity_as_explicit_epoch(self):
        # Original ML.py passes infidelity to every scheduler, so CosineLR
        # interprets each value as an epoch instead of advancing a counter.
        scheduler = make_scheduler("CosineLR")
        for metric in (.9, .5, .1, .5):
            expected = .3 * (1 + np.cos(np.pi * metric / 50)) / 2
            self.assertAlmostEqual(scheduler.step(metric), expected, places=13)


if __name__ == "__main__":
    unittest.main()
