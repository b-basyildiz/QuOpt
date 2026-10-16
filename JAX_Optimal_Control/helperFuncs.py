"""Helpers for the JAX control-flow implementation.

Gate, coupling, drive, and basis definitions retain the PyTorch pipeline's
conventions. Differentiable helpers use JAX arrays with double precision.
The historical integrators retain their original update formulas; this module
is not a change to the physical model or an accuracy correction of those methods.
Unused commented research/plotting code from the original is omitted.
"""
from itertools import permutations, product
from math import ceil

import numpy as np
from numpy import array, kron, ones, zeros
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import expm
import scipy.linalg

def genDrive(d, dTrans,type):
    '''
    DESC: Creates drives based on the dimension and transition wanted. \n

    PARAMS: 
        - d: energy level of the system 
        - dTrans: the transition wanted in the drive (between either dTrans and dTrans - 1) or a two Phonon transition
        - type: Type of drive wanted. Ex: X,Y,Z, or TPX, or TPY. TP stands for two phonon transition. \n

    OUTPUT: d x d numpy matrix with selected transition \n

    AUTHOR: Bora Basyildiz
    '''
    if d < 2:
        raise Exception("The energy levels of our system must be greater than 2. Your input has " + str(d) + " energy levels.")
    if dTrans == 0 or dTrans > d - 1:
        raise Exception("The X gate must be between energy levels withn our system. Right now your transition is in between |" + str(dTrans-1)  + "> -> |" + str(dTrans) + ">.\n Number of energy levels is " + str(d) + ".")
    type = type.lower()
    if type != "x" and type != "y" and type != "tpx" and type != "tpy":
        raise Exception("The drive must be an X, Y, or two Phonon transition (input X, Y, TPX, or TPY).")

    cIndex = 1
    if type[:2] == "tp": 
        cIndex = 2
        type = type[2:] 

    drive = zeros((d,d),dtype=complex)
    val = 0
    if type == "x": val = 1
    else: val = -1j
    drive[dTrans-cIndex,dTrans] = val
    return drive + drive.conj().T

def genWeightHeader(level, leakage=False, N=2):
    '''
    DESC: Column labels for the pulse-weight matrices written by ControlFlow.py/fidelity_ml. \n

    PARAMS:
        - level: energy level of the system (post leakage increment, i.e. the same `level`
          ControlFlow.py passes into fidelity_ml)
        - leakage: whether the leakage model is active. Leakage runs one fewer transition
          in the base drive set (the top transition is handled separately as `ldrives`),
          matching the `for l in range(1,level-1)` vs `for l in range(1,level)` split in
          ControlFlow.py \n
        - N: number of qudits (fixed at 2 in this codebase) \n

    OUTPUT: list of column names, one per column of W, ordered to match sum_pauli's
    `pulse_coef[i*N:(i+1)*N]` slicing in ML.py: for each transition l (1,2,...), X then Y,
    and for each, one column per qudit (q0, q1, ...). More energy levels means more
    transitions and therefore more columns (e.g. quatrit has one more X/Y pair than qutrit).
    Used for every model (square or continuous/time-dependent pulses, with or without
    cross-talk) since they all share this same X/Y-per-transition-per-qudit weight layout -
    ContPulse only changes the time envelope multiplying a given coefficient, not what that
    coefficient represents. \n

    AUTHOR: Bora Basyildiz
    '''
    upperTransition = level - 1 if leakage else level
    cols = []
    for l in range(1, upperTransition):
        for ptype in ("X", "Y"):
            for q in range(N):
                cols.append(f"{ptype}{l-1}{l}_q{q}")
    return cols

def gateGen(gateType,l,d=2):
    '''
    DESC: Creates a two-qubit gate (CNOT, SWAP, iSWAP) for a given energy level \n

    PARAMS: 
        - gateType: string of desired gate
        - l: number of energy levels in the system (qubit is 2) \n
    
    OUTPUT: l^2 x l^2 target numpy matrix with two-qubit gate in qubit-subspace. \n

    AUTHOR: Bora Basyildiz
    '''
    G = zeros((l ** 2, l ** 2),dtype=complex)
    if l < 2: raise Exception("System must have 2 or more energy levels.")
    if gateType == "CNOT":
        for i in range(l ** 2):
            if i == l: G[i,i+1] = 1
            elif i == l+1: G[i,i-1] = 1
            else: G[i,i] = 1
    elif gateType == "CNOT_0":
        for i in range(l ** 2):
            if i == l: G[i,i+1] = 1
            elif i == l+1: G[i,i-1] = 1
            else: 
                if i == 0 or i == 1:
                    G[i,i] = 1
                else:
                    G[i,i] = 0
    elif gateType == "iSWAP":
        G[1,l] = 1j
        G = G + G.T
        for i in range(len(G)):
            if i != 1 and i != l:
                G[i,i] = 1
    elif gateType == "NiSWAP":
        G[1,l] = -1j
        G = G + G.T
        for i in range(len(G)):
            if i != 1 and i != l:
                G[i,i] = 1
    elif gateType == "CZ_0":
        for i in range(l ** 2):
            if i == l+1: G[i,i] = -1
            elif i == 0 or i == 1 or i == l: G[i,i] = 1
            else: G[i,i] = 0
    elif gateType == "CZ": 
        G = gen_CZ(d,l)
    elif gateType == "SWAP":
        G = gen_SWAP(d,l)
    elif gateType == "Toffoli":
        G = np.eye(l**2, dtype=complex)
        idx = (l - 1) * l + (l - 1)
        G[idx, idx] = -1
    else:
        raise Exception("Gate type not implemented.")
    return G

def couplHGen(ctype,l):
    '''
    DESC: Generates coupling Matrix based on coupling type and energy level count. \n

    PARAMS: 
        - ctype: Coupling Matrix Type ('SpeedUp')
        - l: energy level number \n

    OUTPUT: Coupling Matrix (numpy) \n

    AUTHOR: Bora Basyildiz
    '''
    H0 = zeros((l ** 2,l ** 2))
    if ctype == "SpeedUp":
        H0[l+1,0] = 1
        H0[l+1,2] = np.sqrt(2)
        H0[l+1,2*l] = np.sqrt(2)
        H0[l+1,2*l+2] = 2
        H0 = H0 + H0.T
    else:
        raise Exception("Coupling matrix type not implemented. Availible gate types are 'SpeedUp.'")
    return H0

def genCouplMat(couplingType, level):
    H0 = None
    if couplingType == "XX":
        sx = genDrive(level,1,"x")
        H0 = kron(sx,sx) 
    elif couplingType == "XXX":
        sx = array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) 
        sxx = array([[0,0,0],[0,0,1],[0,1,0]])
        H0 = kron(sx,sx) + kron(sxx,sxx)
    elif couplingType == "ZZ":
        if level == 2:
            sz = array([[1,0],[0,-1]])
            id = array([[1,0],[0,1]])
        elif level == 3:
            sz = array([[1,0,0],[0,-1,0],[0,0,0]])
            id = array([[1,0,0],[0,1,0],[0,0,1]])
        H0 = kron(sz,id) + kron(id,sz) + kron(sz,sz)
    elif couplingType == "capacitiveCoup":
        a = np.zeros((level, level))
        for i in range(len(a)-1):
            a[i,i+1] = np.sqrt(i+1)
        H0 = kron(a + a.T,a + a.T)
    elif couplingType == "capacitiveCoupUnit":
        a = np.zeros((level, level))
        for i in range(len(a)-1):
            a[i,i+1] = 1 #unit couplings
        H0 = kron(a + a.T,a + a.T)
    elif couplingType == "capacitiveCoupExtra":
        a = np.zeros((level, level))
        for i in range(len(a)-1):
            a[i,i+1] = i+1
        H0 = kron(a + a.T,a + a.T)
    elif couplingType == "capacitiveCoupMin":
        a = np.zeros((level, level))
        for i in range(len(a)-1):
            a[i,i+1] = 1/(i+1)
        H0 = kron(a + a.T,a + a.T)
    # elif couplingType == "AshhabHopp":
    #     annhilate = array([[0,1,0],[0,0,np.sqrt(2)],[0,0,0]])
    #     create = annhilate.T
    #     H0 = kron(annhilate,create) + kron(create,annhilate)
    # elif couplingType == "AshhabLabFrame":
    #     #Couplings Terms
    #     annhilate = array([[0,1,0],[0,0,np.sqrt(2)],[0,0,0]])
    #     create = annhilate.T
    #     H0 = kron(annhilate + create,annhilate + create)
    #     #Diagonal Entries 
    #     diagEntries = [0, 5.440, 10.681, 4.994, 10.433, 15.666, 9.832, 15.270, 20.506]
    #     for i,d in enumerate(diagEntries):
    #         H0[i,i] = d
    elif couplingType == "CnotProtocol":
        H = np.zeros([3 ** 2, 3 ** 2])
        H[3,4] = 1
        H[3,5] = 1
        H0 = H + H.transpose()
        tmin = np.pi/2
    elif couplingType == "iSwapProtocol":
        H = np.zeros([3 ** 2, 3 ** 2])
        H[1,3] = 1
        H[2,3] = 1
        H0 = H + H.transpose()
    elif couplingType == "SpeedUp":
        H0 = couplHGen(couplingType,level)
    elif couplingType[:8] == "AnalyNeg":
        vals = [1,-1] * 3
        negList = list(permutations(vals,4))
        perm = int(couplingType[8:])
        negs = negList[perm]
        H0 = np.zeros([3 ** 2, 3 ** 2])

        H0[0,4] = negs[0]*1
        H0[2,4] = negs[1]*np.sqrt(2)
        H0[6,4] = negs[2]*np.sqrt(2)
        H0[8,4] = negs[3]*2

        H0 = H0 + H0.transpose()
        couplingType = couplingType[:8]
    elif couplingType == "AllCouplings":
        H0 = np.zeros([3 ** 2, 3 ** 2])
        for i in range(9):
            if i != 4: H0[i,4] = 1
        H0 = H0 + H0.transpose()
    elif couplingType == "Diagonal":
        H0 = np.zeros([3 ** 2, 3 ** 2])
        Diagonal = True
    elif couplingType == "AllCouplingsDiag":
        H0 = np.ones([level ** 2, level ** 2])
        for i in range(level ** 2):
            H0[i,i] = 0
    elif couplingType == "capacitiveCoup_EC":
        a = np.zeros((level, level))
        for i in range(level - 1):
            a[i, i+1] = np.sqrt(i + 1)  
        H0 = kron(a.T, a) + kron(a, a.T)
    else: raise Exception("Incorrect Coupling Type. (XX, Ashabb, AshhabOnes, AshhabHopp, AshhabbLabFrame, CnotProtocol, iSwapProtocol, AnalyticalSpeedUp, AllCouplings, Diagonal, AllCouplingsDiag, capacitiveCoup_EC)")
    return H0

def genEvec(anharm1,anharm2,stag,g,level):
    '''
    DESC: Generates set energy levels for a two qudit system. \n

    PARAMS: 
        - anharm1: Anharmonicity for qudit 1 (in the units of coupling strength) \n
        - anharm2: Anharmonicity for qudit 2 (in the units of coupling strength) \n
        - stag: Staggering between two qudits (in the units of coupling strength) \n
        - g: coupling strength between two qudits \n
        - level: highest energy level for system \n

    OUTPUT: [Vector of Energy levels of qudit 1, Vector of Energy levels of qudit 2, [E11,E22]]

    AUTHOR: Bora Basyildiz 
    
    '''

    E10 = 500/3 #Set qubit intial frequency to 5 GHz. Here we assume that g ~ 30 MHz.
    E01 = E10 - stag*g

    Evec1 = [0,E10] #Energy levels for qubit 1
    for i in range(1,level+1):
        Evec1.append((i+1)*E10+i*anharm1*g)

    Evec2 = [stag*g,E01] #Energy levels for qubit 2
    for i in range(1,level+1):
        Evec2.append((i+1)*E01+i*anharm2*g)

    E11 = E10 + E01
    E22 = Evec1[2] + Evec2[2]
    Evec3 = [E11,E22] # Energy levels for multi-excited states

    # The three vectors have different lengths; keep them separate.
    return tuple(np.asarray(v, dtype=float) for v in (Evec1, Evec2, Evec3))

def printMat(M): #Prints how matrices in more readible fashion
    precision = 2
    for row in M:
        for val in row:
            startstr= " "
            endstr = "+"
            if np.real(val) < 0: startstr = "-"
            if np.imag(val) <0: endstr = "-"
            print(startstr + f"{abs(round(np.real(val),precision)):.{precision}f}",end=endstr) 
            print(f"{abs(round(np.imag(val),precision)):.{precision}f}",end="i,")
        print()

def genXMat(d,l):
    X = np.zeros((l,l),dtype=complex)
    for i in range(d-1): X[i+1,i] = 1
    X[0,d-1] = 1
    return X

def genZMat(d,l):
    Z = np.zeros((l,l),dtype=complex)
    for i in range(d):
        Z[i,i] = np.e ** (2*np.pi*1j*i/d)
    return Z

def genQuditBasis(d,l):
    iterlist = list(product(range(d),repeat=2))
    generators = []
    for tuple in iterlist: generators.append(np.linalg.matrix_power(genXMat(d,l),tuple[0])@np.linalg.matrix_power(genZMat(d,l),tuple[1])) # Need to make X and Z gate generation
    return generators

def gen_CZ(d,l):
    k = 0
    mat = np.eye(l**2,dtype=complex)
    for i in range(l):
        for j in range(l):
            if i <= d and j <= d:
                val = np.e**(2*np.pi*1j/d*i*j)
                mat[k,k] = val
            k += 1
    return mat

def gen_SWAP(d,l):
    '''
    DESC: Two-qudit SWAP gate restricted to the d-dimensional computational subspace,
          acting as identity on anything touching a leakage level (index >= d). When
          d == l (no leakage) this reduces exactly to the plain l-dimensional SWAP. \n

    AUTHOR: Bora Basyildiz
    '''
    mat = np.eye(l**2,dtype=complex)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            mat[i*l+j,i*l+j] = 0
            mat[i*l+j,j*l+i] = 1
    return mat

def dUdt(t, U, H):
    """Schrodinger equation, differentiable with respect to controls in H."""
    return -1j * jnp.matmul(H(t), U)


def normU(U):
    U = jnp.asarray(U)
    return U / jnp.sqrt(jnp.trace(U.conj().T @ U) / len(U))


def normPrint(U):
    U = jnp.asarray(U)
    print(jnp.sqrt(jnp.trace(U.conj().T @ U) / len(U)))


def RK4(t0, tf, U0, h, dUdt, H):
    """Preserve the original normalized-stage RK4 update."""
    U = jnp.asarray(U0)
    for _ in range(ceil((tf - t0) / h)):
        k1 = normU(h * dUdt(t0, U, H))
        k2 = normU(h * dUdt(t0 + 0.5*h, U + 0.5*k1, H))
        k3 = normU(h * dUdt(t0 + 0.5*h, U + 0.5*k2, H))
        k4 = normU(h * dUdt(t0 + h, U + k3, H))
        U = U + (k1 + 2*k2 + 2*k3 + k4) / 6
        t0 += h
    return U


def RK2(t0, tf, U0, h, dUdt, H):
    n = ceil((tf - t0) / h)
    h = (tf - t0) / n
    U = jnp.asarray(U0)
    for _ in range(n):
        k1 = dUdt(t0, U, H)
        k2 = dUdt(t0 + 0.5*h, U + 0.5*h*k1, H)
        U = U + h*k2
        t0 += h
    return U


def SRK2(t0, tf, U0, h, H):
    n = ceil((tf - t0) / h)
    h = (tf - t0) / n
    U = jnp.asarray(U0)
    identity = jnp.eye(len(U), dtype=U.dtype)
    for _ in range(n):
        H1, H2 = H(t0 + 0.25*h), H(t0 + 0.75*h)
        k1 = jnp.linalg.solve(identity + 0.25j*h*H1, -1j*H1 @ U)
        k2 = jnp.linalg.solve(identity + 0.25j*h*H2, -1j*H2 @ U - 0.5j*h*H2 @ k1)
        U = U + 0.5*h*(k1 + k2)
        t0 += h
    return U


def RKN4(t0, tf, U0, h, dUdt, H):
    U = jnp.asarray(U0)
    for _ in range(ceil((tf - t0) / h)):
        k1 = normU(dUdt(t0 + h/2, U, H))
        k2 = normU(dUdt(t0 + h/2, U + h/2*k1, H))
        k3 = normU(dUdt(t0 + h/2, U + h/2*k2, H))
        k4 = normU(dUdt(t0 + h, U + h*k2, H))
        U = U + h/6*(k1 + 2*k2 + 2*k3 + k4)
        t0 += h
    return U


def S(t, H):
    return jnp.imag(H(t))


def K(t, H):
    return jnp.real(H(t))


def fu(U, V, t, H):
    return S(t, H) @ U - K(t, H) @ V


def fv(U, V, t, H):
    return K(t, H) @ U + S(t, H) @ V


def SV2(t0, tf, U0, h, H):
    U, V = jnp.real(U0), -jnp.imag(U0)
    identity = jnp.eye(len(U), dtype=U.dtype)
    for _ in range(ceil((tf - t0) / h)):
        l1 = jnp.linalg.solve(identity - h/2*S(t0 + h/2, H),
                              K(t0 + h/2, H) @ U + S(t0 + h/2, H) @ V)
        V1 = V + h/2*l1
        k1 = fu(U, V1, t0, H)
        k2 = jnp.linalg.solve(identity - h/2*S(t0 + h, H),
                              S(t0 + h, H) @ (U + h/2*k1) - K(t0 + h, H) @ V1)
        U2 = U + h/2*(k1 + k2)
        l2 = fv(U2, V1, t0 + h/2, H)
        U, V = U2, V + h/2*(l1 + l2)
        t0 += h
    return U - 1j*V


def cp(t, coef1, coef2, maxDriveStrength, phase=0):
    """Complex tone amplitude, using the original cosine parameterization."""
    return (maxDriveStrength/np.sqrt(2) *
            (jnp.cos(coef1) + 1j*jnp.cos(coef2)) * jnp.exp(1j*phase*t))


def CTL_drives(t, pc, stag, anharmVal, maxDriveStrength, ContPulse, M, tmin):
    """Preserve the CTL tone ordering and global sin-squared envelope."""
    shape = jnp.sin(jnp.pi*t*M/tmin)**2 if ContPulse == "True" else 1
    D1 = shape*(cp(t, pc[0], pc[4], maxDriveStrength) +
                cp(t, pc[1], pc[5], maxDriveStrength, anharmVal) +
                cp(t, pc[2], pc[6], maxDriveStrength, stag) +
                cp(t, pc[3], pc[7], maxDriveStrength, stag + anharmVal))
    D2 = shape*(cp(t, pc[2], pc[6], maxDriveStrength) +
                cp(t, pc[3], pc[7], maxDriveStrength, anharmVal) +
                cp(t, pc[0], pc[4], maxDriveStrength, -stag) +
                cp(t, pc[1], pc[5], maxDriveStrength, -stag + anharmVal))
    return D1, D2


def stateProj(eLevel, maxLevel, output=False):
    """Constant subspace mask; eLevel and maxLevel must be static integers."""
    psi = np.ones(maxLevel + 1)
    psi[eLevel] = 2
    psi = (np.kron(psi, psi) != 1).astype(np.int32)
    psi[eLevel*(maxLevel + 1) + eLevel + 1:] = 0
    if output:
        for i in range(maxLevel + 1):
            for j in range(maxLevel + 1):
                print(f"|{i}{j}>: {psi[(maxLevel + 1)*i + j]}")
    return jnp.asarray(psi)


def vecSpaceGen(arr):
    """Build constant basis vectors outside jit (the mask determines shape)."""
    mask = np.asarray(arr)
    return jnp.asarray(np.eye(len(mask))[np.flatnonzero(mask)], dtype=jnp.complex128)


def spaceOcc(U, state, stateVecs):
    occVal = 0
    for stateVec in stateVecs:
        occVal += jnp.abs(stateVec.conj().T @ U @ state)**2
    return occVal


def torchMax(values):
    """Legacy public name: differentiable maximum retaining the first tied entry."""
    maximum = values[0]
    for value in values:
        maximum = jnp.where(value > maximum, value, maximum)
    return maximum


def HC(t, Evec, l):
    """Time-dependent coupling, with the same energy and phase conventions."""
    n = l + 1
    E1, E2, E3 = (jnp.asarray(v) for v in Evec)
    ladder = jnp.sqrt(jnp.arange(1, n, dtype=jnp.float64))
    a1 = jnp.diag(ladder * jnp.exp(-1j*(E1[:n-1] - E1[1:n])*t), 1)
    a2 = jnp.diag(ladder * jnp.exp(-1j*(E2[:n-1] - E2[1:n])*t), 1)
    resonant = 2*(jnp.cos(-E3[0]*t) + jnp.cos((E2[2] - E3[0])*t) +
                  jnp.cos((E1[2] - E3[0])*t) + jnp.cos((E3[1] - E3[0])*t))
    return resonant*jnp.kron(a1 + a1.conj().T, a2 + a2.conj().T)


def genTwoQuditBasis(d, l, dt=jnp.complex128):
    generators = genQuditBasis(d, l)
    identity = np.diag(np.arange(l) < d)
    basis = []
    for i, j in product(range(len(generators)), repeat=2):
        left = identity if i == 0 else generators[i]
        right = identity if j == 0 else generators[j]
        basis.append(jnp.asarray(np.kron(left, right), dtype=dt))
    return basis


def PWC(t0, tf, H, U0, n, order):
    """Piecewise-constant propagation, retaining NumPy-input midpoint semantics."""
    h = (tf - t0)/n
    U = U0
    if isinstance(U0, np.ndarray):
        for _ in range(n):
            U = scipy.linalg.expm(-1j*h*H(t0 + h/2)) @ U
            t0 += h
        return U
    U = jnp.asarray(U0)
    for _ in range(n):
        if order.casefold() == "first":
            U = expm(-1j*h*H(t0)) @ U
        elif order.casefold() == "second":
            U = expm(-1j*h*H(t0 + h/2)) @ U
        t0 += h
    return U
