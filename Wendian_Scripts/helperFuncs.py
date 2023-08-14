'''
DESC: This file helperFuncs.py provides helper functions for the fullControl.py and Full_ML.py files.
INPUTS: N/A
OUTPUTS: N/A
AUTHOR: Bora Basyildiz
'''
#Imports
from numpy import zeros, array
import numpy as np
from torch import tensor, matmul, cdouble, trace, sqrt

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
    
def RK4(t0, tf, U0, h, dUdt,H):
    '''
    DESC: Implements complex ODE solver for pyTorch Matrices with Runge-Kutta fourth order method. \n

    PARAMS: 
        - t0: start time 
        - tf: end time 
        - U0: Initial unitary Matrix
        - h: change in time for each step 
        - dUdt: Schrödinger equation 
        = H: Time dependent Hamiltonian \n

    OUTPUT: pyTorch Tensor with ODE solved over time intervals (note gate may not be unitary)

    AUTHOR: Bora Basyildiz & Prateek Bhindwar
    
    '''
    # number of iterations based on dt(h)
    n = (int)((tf- t0)/h)
    U = U0
    for i in range(n):
        #RK4 method
        k1 = h * dUdt(t0, U, H)
        k2 = h * dUdt(t0 + 0.5 * h, U + 0.5 * k1, H)
        k3 = h * dUdt(t0 + 0.5 * h, U + 0.5 * k2, H)
        k4 = h * dUdt(t0 + h, U + k3, H)
 
        # Updating U and time-step
        U = U + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        t0 = t0 + h
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
    return -1j*matmul(H(t),U)

def normU(U):
    '''
    DESC: Normalizes Unitary Matrix \n
    
    PARAMS:
        - U: Unitary Matrix (pyTorch tensor) \n
    
    OUTPUT: Normalized Unitary Matrix \n

    AUTHOR: Bora Basyildiz
    '''
    norm = sqrt(trace(matmul(U.conj().T,U))/len(U))
    return U/norm

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