'''
DESC: This file helperFuncs.py provides helper functions for the fullControl.py and Full_ML.py files.
INPUTS: N/A
OUTPUTS: N/A
AUTHOR: Bora Basyildiz
'''
#Imports
from numpy import zeros, array, kron
import numpy as np
from torch import tensor, matmul, cdouble, trace, sqrt
from itertools import permutations
import torch 
from math import ceil
import scipy

#Function definitions
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

def gateGen(gateType,l):
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
    elif gateType == "iSWAP":
        G[1,l] = 1j
        G = G + G.T
        for i in range(len(G)):
            if i != 1 and i != l:
                G[i,i] = 1
    else:
        raise Exception("Gate type not implemented.")
    return G
    
def RK4(t0, tf, U0, h, dUdt, H):
    '''
    DESC: Implements complex ODE solver for pyTorch Matrices with Runge-Kutta fourth order method. \n

    PARAMS: 
        - t0: start time 
        - tf: end time 
        - U0: Initial unitary Matrix
        - h: change in time for each step 
        - dUdt: Schrödinger equation 
        - H: Time dependent Hamiltonian \n

    OUTPUT: pyTorch Tensor with ODE solved over time intervals (note gate may not be unitary) \n

    AUTHOR: Bora Basyildiz & Prateek Bhindwar
    
    '''
    # number of iterations based on dt(h)
    n = ceil((tf-t0)/h)
    U = U0
    for i in range(n):
        #RK4 method
        k1 = normU(h * dUdt(t0, U, H))
        k2 = normU(h * dUdt(t0 + 0.5 * h, U + 0.5 * k1, H))
        k3 = normU(h * dUdt(t0 + 0.5 * h, U + 0.5 * k2, H))
        k4 = normU(h * dUdt(t0 + h, U + k3, H))
 
        # Updating U and time-step
        U = U + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        t0 = t0 + h
    return U

def RK2(t0, tf, U0, h, dUdt, H):
    n = ceil((tf-t0)/h)
    U = U0
    t = t0

    for i in range(n):
        #k1 = dUdt(t, U, H)
        #k2 = dUdt(t + 0.5*h, U + 0.5*k1, H)
        k1 = dUdt(t, U, H)
        k2 = dUdt(t + 0.5*h, U + 0.5*h*k1, H)

        #U = U + h/6 * (k1 + 2*k2)
        U = U + h*k2
        t = t + h
    return U

def dUdt(t, U, H):
    '''
    DESC: Schrödinger equation in ODE form for unitaries. Necessary for RK4 method. \n

    PARAMS:
        - t: time the ODE is being evaluated
        - U: Unitary being evolved. 
        - H: Time dependent Hamiltonian \n

    OUTPUT: Matrix to be used for RK4 method \n

    AUTHOR: Bora Basyildiz
    '''
    if isinstance(U,torch.Tensor):
        return -1j*matmul(H(t),U)
    else:
        return -1j*np.matmul(H(t),U)

def SRK2(t0, tf, U0, h, H):
    n = ceil((tf-t0)/h)
    U = U0
    t = t0
    I = tensor(np.eye(len(U)))

    for i in range(n):
        k1 = torch.linalg.solve(I + 1j*h*0.25*H(t + 0.25*h), -1j*H(t+0.25*h) @ U)
        k2 = torch.linalg.solve(I + 1j*h*0.25*H(t + 0.75*h), -1j*H(t+ 0.75*h) @ U - 1j*0.5*h*H(t+0.75*h) @ k1)

        U = U + 0.5*h*(k1 + k2)
        t = t + h
    return U


def RKN4(t0, tf, U0, h, dUdt, H):
    '''
    DESC: Implements complex ODE solver for pyTorch Matrices with Symplectic Runge-Kutta fourth order method. \n
    
    PARAMS: 
        - t0: start time 
        - tf: end time 
        - U0: Initial unitary Matrix
        - h: change in time for each step 
        - dUdt: Schrödinger equation 
        - H: Time dependent Hamiltonian \n

    OUTPUT: pyTorch Tensor with ODE solved over time intervals (note gate may not be unitary) \n

    AUTHRO: Bora Basyildiz & Will Beason
    '''
    n = ceil((tf-t0)/h)
    U = U0
    t = t0
    for i in range(n):
        k1 = normU(dUdt(t + h/2,U, H))
        k2 = normU(dUdt(t + h/2, U + h/2*k1, H))
        k3 = normU(dUdt(t + h/2, U + h/2*k2, H))
        k4 = normU(dUdt(t + h, U + h*k2, H))

        U = U + h/6 * (k1+2*k2 + 2*k3 + k4)
        t = t + h
    return U

def SV2(t0, tf, U0, h, H):
    '''
    DESC: Störmer-Verlet symplectic ODE Solver (Second Order). \n

    PARAMS: 
        - t0: start time 
        - tf: end time 
        - U0: Initial unitary Matrix
        - h: change in time for each step 
        - H: Time dependent Hamiltonian \n

    OUTPUTS: pyTorch Tensor with ODE solved over time intervals (note gate may not be unitary) \n

    AUTHOR: Bora Basyildiz
    '''
    n = ceil((tf-t0)/h)
    tn = t0
    if isinstance(U0,torch.Tensor):
        Un = torch.real(U0)
        Vn = -torch.imag(U0)
        I = tensor(np.eye(len(Un)))
    else:
        Un = np.real(U0)
        Vn = -np.imag(U0)
        I = np.eye(len(Un))

    for i in range(n):
        # Substep and subcall generating
        U1 = Un
        if isinstance(U0,torch.Tensor):
            l1 = torch.linalg.solve(I - (h/2) * S(tn + h/2, H), K(tn + h/2, H) @ U1 + S(tn + h/2, H) @ Vn)
        else:
            l1 = scipy.linalg.solve(I - (h/2) * S(tn + h/2, H), K(tn + h/2, H) @ U1 + S(tn + h/2, H) @ Vn)
        V1 = Vn + (h/2) * l1
        V2 = V1
        k1 = fu(U1, V1, tn, H)
        if isinstance(U0,torch.Tensor):
            k2 = torch.linalg.solve(I - (h/2) * S(tn + h, H), S(tn + h, H) @ (Un + (h/2) * k1) - K(tn + h, H) @ V1)
        else:
            k2 = scipy.linalg.solve(I - (h/2) * S(tn + h, H), S(tn + h, H) @ (Un + (h/2) * k1) - K(tn + h, H) @ V1)
        U2 = Un + (h/2) * (k1 + k2)
        l2 = fv(U2, V2, tn + h/2, H)

        # Time-step Evolution
        Un = Un + (h/2) * (k1 + k2)
        Vn = Vn + (h/2) * (l1 + l2)
        tn = tn + h
    return Un - 1j*Vn
    
def S(t, H):
     if isinstance(H,torch.Tensor):
        return torch.imag(H(t))
     else:
        return np.imag(H(t))

def K(t, H):
    if isinstance(H,torch.Tensor):
        return torch.real(H(t))
    else:
        return np.real(H(t))
    
def fu(U, V, t, H):
    return S(t, H) @ U - K(t, H) @ V 

def fv(U, V, t, H):
    return K(t, H) @ U + S(t, H) @ V

def normU(U):
    '''
    DESC: Normalizes Unitary Matrix \n
    
    PARAMS:
        - U: Unitary Matrix (pyTorch tensor) \n
    
    OUTPUT: Normalized Unitary Matrix \n

    AUTHOR: Bora Basyildiz
    '''
    if isinstance(U,torch.Tensor):
        norm = sqrt(trace(matmul(U.conj().T,U))/len(U))
    else: 
        norm = np.sqrt(np.trace(np.matmul(U.conj().T,U))/len(U))
    return U/norm

def normPrint(U):
    if isinstance(U,torch.Tensor):
        print(sqrt(trace(matmul(U.conj().T,U))/len(U)))
    else:
        print(np.sqrt(np.trace(np.matmul(U.conj().T,U))/len(U)))

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
    elif couplingType == "Ashhab":
        annhilate = array([[0,1,0],[0,0,np.sqrt(2)],[0,0,0]])
        create = annhilate.T
        H0 = kron(annhilate + create,annhilate + create)
    elif couplingType == "AshhUnit":
        annhilate = array([[0,1,0],[0,0,1],[0,0,0]])
        create = annhilate.T
        H0 = kron(annhilate + create,annhilate + create)
    elif couplingType == "AshhabHopp":
        annhilate = array([[0,1,0],[0,0,np.sqrt(2)],[0,0,0]])
        create = annhilate.T
        H0 = kron(annhilate,create) + kron(create,annhilate)
    elif couplingType == "AshhabLabFrame":
        #Couplings Terms
        annhilate = array([[0,1,0],[0,0,np.sqrt(2)],[0,0,0]])
        create = annhilate.T
        H0 = kron(annhilate + create,annhilate + create)
        #Diagonal Entries 
        diagEntries = [0, 5.440, 10.681, 4.994, 10.433, 15.666, 9.832, 15.270, 20.506]
        for i,d in enumerate(diagEntries):
            H0[i,i] = d
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
        H0[2,4] = negs[1]*sqrt(2)
        H0[6,4] = negs[2]*sqrt(2)
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
    else: raise Exception("Incorrect Coupling Type. (XX, Ashabb, AshhabOnes, AshhabHopp, AshhabbLabFrame, CnotProtocol, iSwapProtocol, AnalyticalSpeedUp, AllCouplings, Diagonal, AllCouplingsDiag)")
    return H0

# def cp(t,coef,mDS,M,tmin,phase=0):
#     '''
#     DESC: Generates continuous pulse based on parameters and phase

#     PARAMS: 
#         - t: time
#         - coef: torch coefficient
#         - phase: time dependent phase
#         - mDS: maximum drive strength
#         - M: total pulse segment count
#         - tmin: Speed limit for given numerics

#     OUTPUT: Function for time-dependent pulse segment

#     AUTHOR: Bora Basyildiz
#     '''
#     c = mDS*torch.cos(coef)
#     p = tensor(1j*phase*t)
#     shape = tensor((np.sin(np.pi * t * M / tmin)) ** 2)
#     return c*p*shape
