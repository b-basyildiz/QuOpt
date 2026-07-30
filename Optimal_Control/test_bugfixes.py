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

      3. CTL_drives (helperFuncs.py, extracted out of ML.py's CTL_H) builds the D1/D2 multi-tone
         drive amplitudes for the "Qutrit CTL Model" (paper Sec. 3): D_1^m = W11 + W12*e^{i*d*t} +
         W21*e^{i*w*t} + W22*e^{i*(d+w)*t}, D_2^m = W21 + W22*e^{i*d*t} + W11*e^{-i*w*t} +
         W12*e^{i*(d-w)*t}. D2 had two of its four terms swapped: the coefficient pair meant for
         qudit 2's own (resonant, phase-0) 0<->1 drive (W21) and the pair meant for the crosstalk
         leak from qudit 1's 1<->2 drive (W12, phase d-w) were exchanged. This left qudit 2 with no
         genuine resonant own-transition drive at all (that slot held the misplaced W12 instead),
         letting the optimizer dodge the mandatory crosstalk coupling by suppressing W12 -- which
         both hid crosstalk's effect (inflating fidelity) and left the physics wrong at long T.

AUTHOR: Claude, at Bora Basyildiz's request
'''
import inspect
import unittest

import numpy as np
import scipy.linalg
import torch

from helperFuncs import RK2, SRK2, dUdt, genXMat, gateGen, genTwoQuditBasis, cp, CTL_drives, gen_SWAP
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

    def test_ML_module_no_longer_overwrites_anharmVal_from_anharm_matrix(self):
        # anharmVal used to be silently overwritten with anharm[-1,-1], which for level=4
        # (qutrit+leakage) evaluates to 2x the intended base anharmonicity constant (28 vs 14),
        # because anharm's diagonal is built with a different (linear, already-grown-per-level)
        # convention than CTL_H's own quadratic l(l-1)/2 growth formula expects as its input.
        source = inspect.getsource(ML.fidelity_ml)
        self.assertNotIn("anharmVal = float(anharm", source)

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


class TestCTLDrives(unittest.TestCase):
    '''D1/D2 must match the paper's Qutrit CTL Model (Sec. 3) equations term-by-term.

    pc index <-> paper Omega_{i,j} mapping, established by matching D1:
        (pc[0],pc[4]) = W11 (qudit1, own 0<->1)   (pc[1],pc[5]) = W12 (qudit1, own 1<->2)
        (pc[2],pc[6]) = W21 (qudit2, own 0<->1)   (pc[3],pc[7]) = W22 (qudit2, own 1<->2)
    '''

    def setUp(self):
        self.pc = torch.rand(8, dtype=torch.double) * 2 * np.pi
        self.stag = 17.0
        self.anharmVal = 14.0
        self.maxDriveStrength = 40
        self.t = 0.37  # arbitrary, non-special time

    def expected_D1(self, pc):
        w11 = cp(self.t, pc[0], pc[4], self.maxDriveStrength)
        w12 = cp(self.t, pc[1], pc[5], self.maxDriveStrength, self.anharmVal)
        w21 = cp(self.t, pc[2], pc[6], self.maxDriveStrength, self.stag)
        w22 = cp(self.t, pc[3], pc[7], self.maxDriveStrength, self.stag + self.anharmVal)
        return w11 + w12 + w21 + w22

    def expected_D2(self, pc):
        # paper: D2 = W21 + W22*e^{i*d*t} + W11*e^{-i*w*t} + W12*e^{i*(d-w)*t}
        w21 = cp(self.t, pc[2], pc[6], self.maxDriveStrength)
        w22 = cp(self.t, pc[3], pc[7], self.maxDriveStrength, self.anharmVal)
        w11 = cp(self.t, pc[0], pc[4], self.maxDriveStrength, -1 * self.stag)
        w12 = cp(self.t, pc[1], pc[5], self.maxDriveStrength, self.anharmVal - self.stag)
        return w21 + w22 + w11 + w12

    def test_D1_matches_paper_formula(self):
        D1, _ = CTL_drives(self.t, self.pc, self.stag, self.anharmVal, self.maxDriveStrength, "False", 40, 1.0)
        self.assertAlmostEqual(abs(D1 - self.expected_D1(self.pc)).item(), 0.0, places=10)

    def test_D2_matches_paper_formula(self):
        _, D2 = CTL_drives(self.t, self.pc, self.stag, self.anharmVal, self.maxDriveStrength, "False", 40, 1.0)
        self.assertAlmostEqual(abs(D2 - self.expected_D2(self.pc)).item(), 0.0, places=10)

    def test_D2_is_not_the_old_swapped_formula(self):
        # Regression guard: the old (buggy) D2 swapped (pc[1],pc[5]) <-> (pc[2],pc[6]).
        _, D2 = CTL_drives(self.t, self.pc, self.stag, self.anharmVal, self.maxDriveStrength, "False", 40, 1.0)
        pc = self.pc
        old_buggy_D2 = (cp(self.t, pc[1], pc[5], self.maxDriveStrength)
                         + cp(self.t, pc[3], pc[7], self.maxDriveStrength, self.anharmVal)
                         + cp(self.t, pc[0], pc[4], self.maxDriveStrength, -1 * self.stag)
                         + cp(self.t, pc[2], pc[6], self.maxDriveStrength, -1 * self.stag + self.anharmVal))
        self.assertGreater(abs(D2 - old_buggy_D2).item(), 1.0)

    def test_D2_is_D1_under_qudit_relabeling(self):
        # Physical symmetry: relabeling which qudit is "1" vs "2" (swap W1j<->W2j) and flipping
        # the sign of the staggering should turn D1 into D2. This holds only if D1/D2 are each
        # internally self-consistent (not just individually matching the paper by coincidence).
        pc = self.pc
        pc_swapped = torch.stack([pc[2], pc[3], pc[0], pc[1], pc[6], pc[7], pc[4], pc[5]])
        D1_swapped, _ = CTL_drives(self.t, pc_swapped, -1 * self.stag, self.anharmVal, self.maxDriveStrength, "False", 40, 1.0)
        _, D2 = CTL_drives(self.t, pc, self.stag, self.anharmVal, self.maxDriveStrength, "False", 40, 1.0)
        self.assertAlmostEqual(abs(D1_swapped - D2).item(), 0.0, places=10)


class TestLeakageAwareSWAP(unittest.TestCase):
    '''gen_SWAP(d,l) must swap only the d-dimensional computational subspace, leaving
    anything touching a leakage level (index >= d) untouched, while reducing exactly
    to the plain l-dimensional SWAP when there's no leakage level (d == l).'''

    def test_no_leakage_matches_plain_swap(self):
        for l in [2, 3, 4]:
            G = gen_SWAP(l, l)
            expected = np.zeros((l**2, l**2), dtype=complex)
            for i in range(l):
                for j in range(l):
                    expected[i*l+j, j*l+i] = 1
            self.assertTrue(np.allclose(G, expected), f"mismatch at l={l}")

    def test_leakage_case_is_unitary(self):
        d, l = 4, 5  # ququart computational space + 1 leakage level
        G = gen_SWAP(d, l)
        self.assertTrue(np.allclose(G @ G.conj().T, np.eye(l**2)))

    def test_leakage_level_is_untouched(self):
        d, l = 4, 5
        G = gen_SWAP(d, l)
        I = np.eye(l**2)
        for i in range(l):
            for j in range(l):
                if i >= d or j >= d:
                    idx = i*l+j
                    self.assertTrue(np.allclose(G[idx], I[idx]),
                                     f"row for index ({i},{j}) touching the leak level should be untouched identity")

    def test_computational_subspace_matches_plain_swap(self):
        d, l = 4, 5
        G = gen_SWAP(d, l)
        comp_idx = [i*l+j for i in range(d) for j in range(d)]
        V = G[np.ix_(comp_idx, comp_idx)]
        plain = gen_SWAP(d, d)  # d==l case reduces to plain d-dim SWAP
        self.assertTrue(np.allclose(V, plain))

    def test_gateGen_SWAP_uses_gen_SWAP(self):
        d, l = 4, 5
        self.assertTrue(np.allclose(gateGen("SWAP", l, d), gen_SWAP(d, l)))


if __name__ == "__main__":
    unittest.main()
