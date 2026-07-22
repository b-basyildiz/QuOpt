'''
DESC: Regression tests for two bugs found while debugging the CTL (cross-talk + leakage)
      qutrit CZ data (qutrit_CZ_capacitiveCoup_..._leakage14.0_..._CTh0.01_stag17.0_...):

      1. RK2/SRK2 (helperFuncs.py) computed the number of sub-steps n = ceil((tf-t0)/h) but
         then advanced time using the fixed h instead of the exact (tf-t0)/n, so whenever a
         segment's duration wasn't an exact multiple of h the integrator overshot tf. This was
         most severe for short gate times (segment length << h), since h was fixed (set by the
         CTh command line argument) while the segment length shrinks with T.

      2. ML.py's leakage-aware fidelity used genTwoQuditBasis(dspaceLen, level, dt) with
         dspaceLen = level-1 (the qutrit computational dimension) and level = physical dimension
         (qutrit + 1 leakage level). The resulting generalized Pauli generators are built from
         genXMat/genZMat(dspaceLen, level), which are rank-deficient (non-unitary) when
         dspaceLen < level, and the stand-in "identity" (idTemp) is actually a projector onto the
         computational subspace, not the true identity. The standard average-gate-fidelity trace
         formula requires a genuine unitary operator basis and a genuine identity, so this
         produced an invalid, leakage-insensitive metric. The fix uses genTwoQuditBasis(level,
         level, dt), i.e. a proper operator basis on the full physical Hilbert space, with the
         target gate already embedded as identity on the leakage level (via gateGen).

AUTHOR: Claude, at Bora Basyildiz's request
'''
import inspect
import unittest

import numpy as np
import scipy.linalg
import torch

from helperFuncs import RK2, SRK2, dUdt, genXMat, gateGen, genTwoQuditBasis
import ML


def randomHermitian(dim, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return torch.tensor(scale * (A + A.conj().T), dtype=torch.cdouble)


class TestODEStepSize(unittest.TestCase):
    '''RK2/SRK2 must evolve for exactly (tf-t0), not n*h, when h doesn't evenly divide the interval.'''

    def check_RK2(self, t0, tf, h, H0):
        H = lambda t: H0
        U0 = torch.eye(len(H0), dtype=torch.cdouble)
        U = RK2(t0, tf, U0, h, dUdt, H)
        U_exact = torch.tensor(scipy.linalg.expm(-1j * H0.numpy() * (tf - t0)), dtype=torch.cdouble)
        return torch.linalg.norm(U - U_exact).item()

    def check_SRK2(self, t0, tf, h, H0):
        H = lambda t: H0
        U0 = torch.eye(len(H0), dtype=torch.cdouble)
        U = SRK2(t0, tf, U0, h, H)
        U_exact = torch.tensor(scipy.linalg.expm(-1j * H0.numpy() * (tf - t0)), dtype=torch.cdouble)
        return torch.linalg.norm(U - U_exact).item()

    def test_RK2_short_segment_matches_exact_solution(self):
        # Mirrors the real failure case: T=0.1, M=40, tmin=pi/4 => segment ~=0.00196, h=0.01 (n=1)
        H0 = randomHermitian(2, seed=1)
        t0, tf, h = 0.0, 0.1 * (np.pi / 4) / 40, 0.01
        err = self.check_RK2(t0, tf, h, H0)
        self.assertLess(err, 1e-6, "RK2 should closely match the exact solution over the true segment length")

    def test_RK2_short_segment_no_longer_overshoots(self):
        # The old code effectively evolved for h instead of (tf-t0); verify we no longer match that wrong answer.
        # scale=40 matches the maxDriveStrength magnitude used in the real leakage/crosstalk runs.
        H0 = randomHermitian(2, seed=1, scale=40)
        t0, tf, h = 0.0, 0.1 * (np.pi / 4) / 40, 0.01
        U0 = torch.eye(2, dtype=torch.cdouble)
        U = RK2(t0, tf, U0, h, dUdt, lambda t: H0)
        U_overshot = torch.tensor(scipy.linalg.expm(-1j * H0.numpy() * h), dtype=torch.cdouble)
        self.assertGreater(torch.linalg.norm(U - U_overshot).item(), 0.1,
                            "Fixed RK2 output should differ substantially from the old, overshot-time result")

    def test_RK2_multi_step_segment_matches_exact_solution(self):
        # Segment length not an exact multiple of h (T=2.0 case): n=4 steps, still must land exactly on tf.
        H0 = randomHermitian(2, seed=2)
        t0, tf, h = 0.0, 2.0 * (np.pi / 4) / 40, 0.01
        err = self.check_RK2(t0, tf, h, H0)
        self.assertLess(err, 1e-3, "RK2 should match the exact solution to 2nd-order accuracy")

    def test_SRK2_short_segment_matches_exact_solution(self):
        H0 = randomHermitian(2, seed=3)
        t0, tf, h = 0.0, 0.1 * (np.pi / 4) / 40, 0.01
        err = self.check_SRK2(t0, tf, h, H0)
        self.assertLess(err, 1e-6, "SRK2 should closely match the exact solution over the true segment length")

    def test_SRK2_multi_step_segment_matches_exact_solution(self):
        H0 = randomHermitian(2, seed=4)
        t0, tf, h = 0.0, 2.0 * (np.pi / 4) / 40, 0.01
        err = self.check_SRK2(t0, tf, h, H0)
        self.assertLess(err, 1e-3, "SRK2 should match the exact solution to 2nd-order accuracy")


class TestLeakageFidelity(unittest.TestCase):
    '''genTwoQuditBasis(level, level, dt) must be used (not dspaceLen, level) for leakage runs.'''

    def setUp(self):
        self.level = 4  # qutrit (d=3) computational levels + 1 leakage level
        self.N = 2
        self.dt = torch.cdouble
        self.input_gate = torch.tensor(gateGen("CZ", self.level, self.level - 1), dtype=self.dt)

    def fidelity(self, U_Exp, SU, d2):
        fid = 0
        for U in SU:
            eps_U = torch.matmul(torch.matmul(U_Exp, U), U_Exp.conj().T)
            target_U = torch.matmul(torch.matmul(self.input_gate, U.conj().T), self.input_gate.conj().T)
            fid = fid + torch.trace(torch.matmul(target_U, eps_U))
        return abs((fid + d2 ** 2) / (d2 ** 2 * (d2 + 1))).item()

    def test_old_leakage_dimension_generators_were_not_unitary(self):
        X_old = genXMat(self.level - 1, self.level)
        self.assertGreater(np.linalg.norm(X_old @ X_old.conj().T - np.eye(self.level)), 0.5,
                            "genXMat(dspaceLen, level) with dspaceLen<level is rank-deficient (documents the bug's root cause)")

    def test_fixed_call_uses_unitary_generators(self):
        X_new = genXMat(self.level, self.level)
        self.assertAlmostEqual(np.linalg.norm(X_new @ X_new.conj().T - np.eye(self.level)), 0.0, places=10)

    def test_fixed_basis_identity_element_is_true_identity(self):
        SU = genTwoQuditBasis(self.level, self.level, self.dt)
        identity_element = SU[0]  # tuple (0,0) -> kron(idTemp, idTemp)
        expected = torch.eye(self.level ** self.N, dtype=self.dt)
        self.assertAlmostEqual(torch.linalg.norm(identity_element - expected).item(), 0.0, places=10)

    def test_ML_module_no_longer_uses_dspaceLen(self):
        params = list(inspect.signature(ML.fidelity_ml).parameters)
        self.assertNotIn("dspaceLen", params, "dspaceLen parameter should have been removed from fidelity_ml")
        source = inspect.getsource(ML.fidelity_ml)
        self.assertIn("genTwoQuditBasis(level,level,dt)", source)
        self.assertIn("d2 = level**N", source)

    def test_perfect_implementation_has_unit_fidelity(self):
        SU = genTwoQuditBasis(self.level, self.level, self.dt)
        d2 = self.level ** self.N
        fid = self.fidelity(self.input_gate, SU, d2)
        self.assertAlmostEqual(fid, 1.0, places=9)

    def test_fixed_formula_penalizes_leakage_more_than_old_formula(self):
        # Unitary that fully swaps amplitude out of the top computational level into the leakage level.
        L = torch.eye(self.level, dtype=self.dt)
        L[2, 2] = 0; L[2, 3] = 1
        L[3, 2] = 1; L[3, 3] = 0
        leakOp = torch.kron(L, torch.eye(self.level, dtype=self.dt))
        U_leaky = leakOp @ self.input_gate

        SU_old = genTwoQuditBasis(self.level - 1, self.level, self.dt)  # what ML.py used to call
        SU_new = genTwoQuditBasis(self.level, self.level, self.dt)      # what ML.py calls now

        fid_old = self.fidelity(U_leaky, SU_old, (self.level - 1) ** self.N)
        fid_new = self.fidelity(U_leaky, SU_new, self.level ** self.N)

        self.assertLess(fid_new, 1.0)
        self.assertLess(fid_new, fid_old,
                         "the old projector-based formula under-penalizes leakage relative to the fixed formula")


if __name__ == "__main__":
    unittest.main()
