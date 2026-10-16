"""Regression tests for the Hamiltonian passed to continuous-pulse evolution."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

import ML
from helperFuncs import genDrive


class _HamiltoniansCaptured(Exception):
    """Stop before optimization after inspecting every pulse segment."""


class TestContinuousLeakageHamiltonian(unittest.TestCase):
    def setUp(self):
        self.level = 4  # Three computational levels and one leakage level.
        self.M = 3
        self.duration = 0.6
        self.fractions = (0.0, 0.25, 0.5, 1.0)
        self.weights = np.linspace(0.1, 2.4, self.M * 8).reshape(self.M, 8)
        self.drives = [
            genDrive(self.level, transition, axis)
            for transition in (1, 2) for axis in ("x", "y")
        ]
        self.leakage_drives = [genDrive(self.level, 3, axis) for axis in ("x", "y")]
        self.anharm = np.diag([0.0, 0.0, 14.0, 28.0])
        identity = np.eye(self.level)
        self.drift = np.kron(identity, self.anharm) + np.kron(self.anharm, identity)
        rng = np.random.default_rng(23)
        coupling = rng.normal(size=(self.level**2, self.level**2))
        self.H0 = (coupling + coupling.T) / 2

    def capture_hamiltonians(self, cap):
        captured = []

        def inspect_segment(generator, start, duration, step, order, exponentials):
            # The CFME integrator receives -i H(t); inspect it synchronously while
            # the current segment's coefficients are in scope.
            captured.append([
                (1j * generator(start + fraction * duration)).detach().numpy()
                for fraction in self.fractions
            ])
            if len(captured) == self.M:
                raise _HamiltoniansCaptured
            return torch.eye(self.level**2, dtype=torch.cdouble)

        with patch.object(ML.pd, "read_csv", return_value=ML.pd.DataFrame(self.weights)), \
                patch.object(ML, "generate_cfme_unitary", side_effect=inspect_segment), \
                self.assertRaises(_HamiltoniansCaptured):
            ML.fidelity_ml(
                M=self.M, input_gate=np.eye(self.level**2), tmin=self.duration,
                N_iter=1, rseed=7, H0=self.H0,
                drives=[self.drives, self.leakage_drives, self.anharm],
                maxDriveStrength=cap, lbool=True, minLeak="False",
                crossTalk="False", h=0.01, alpha=0.0, anharmVal=14.0,
                stag=17.0, ode="CFME4", ContPulse="True", optimizer="ADAM",
                weightFName="captured_weights.csv", wsmBoolF=False,
            )
        self.assertEqual(len(captured), self.M)
        return captured

    def expected_drive(self, segment, cap):
        # The upper computational-transition controls also drive the leakage
        # transition in the ordinary leakage model.
        gates = [self.drives[0], self.drives[1],
                 self.drives[2] + self.leakage_drives[0],
                 self.drives[3] + self.leakage_drives[1]]
        amplitudes = self.weights[segment]
        if cap != -1:
            amplitudes = cap * np.cos(amplitudes)
        identity = np.eye(self.level)
        drive = np.zeros_like(self.H0, dtype=complex)
        for index, gate in enumerate(gates):
            drive += amplitudes[2 * index] * np.kron(gate, identity)
            drive += amplitudes[2 * index + 1] * np.kron(identity, gate)
        return drive

    def test_anharmonicity_is_static_for_bounded_and_unlimited_drives(self):
        for cap in (-1, 40):
            hamiltonians = self.capture_hamiltonians(cap)
            for segment, samples in enumerate(hamiltonians):
                for fraction, hamiltonian in zip(self.fractions, samples):
                    with self.subTest(cap=cap, segment=segment, fraction=fraction):
                        # All drives are off-diagonal, so these energies must be
                        # independent of envelope, segment weights, and drive cap.
                        np.testing.assert_allclose(
                            np.diag(hamiltonian - self.H0), np.diag(self.drift),
                            atol=1e-12, rtol=1e-12,
                        )

    def test_computational_and_leakage_drives_keep_sine_squared_envelope(self):
        for cap in (-1, 40):
            hamiltonians = self.capture_hamiltonians(cap)
            for segment, samples in enumerate(hamiltonians):
                drive = self.expected_drive(segment, cap)
                for fraction, hamiltonian in zip(self.fractions, samples):
                    with self.subTest(cap=cap, segment=segment, fraction=fraction):
                        shaped_drive = np.sin(np.pi * fraction)**2 * drive
                        actual = hamiltonian - self.H0
                        # Check the applied fields separately from static energy
                        # shifts, including pulse boundaries and interior samples.
                        np.fill_diagonal(actual, 0)
                        np.testing.assert_allclose(actual, shaped_drive, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
