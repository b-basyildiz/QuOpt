'''
DESC: This file helperFuncs.py provides helper functions for the fullControl.py and Full_ML.py files.
INPUTS: N/A
OUTPUTS: N/A
AUTHOR: Bora Basyildiz
'''
#Imports
from numpy import zeros, array, kron, ones
import numpy as np
from torch import tensor, matmul, cdouble, trace, sqrt
from itertools import permutations,product
import torch 
from math import ceil
import scipy
import matplotlib.pyplot as plt

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
    elif gateType == "CZ":
        for i in range(len(G)):
            if i == l+1: G[i,i] = -1
            else: G[i,i] = 1
    elif gateType == "CZ_0":
        for i in range(l ** 2):
            if i == l+1: G[i,i] = -1
            elif i == 0 or i == 1 or i == l: G[i,i] = 1
            else: G[i,i] = 0
    elif gateType == "CZZ":
        if l < 3:
            raise Exception("CZZ is a qutrit gate. Please have at least three energy levels in your system.")
        for i in range(len(G)):
            if i == l+1: G[i,i] = np.e ** (2*np.pi*1j/3)
            elif i == l+2: G[i,i] = np.e ** (-2*np.pi*1j/3)
            elif i == 2*l + 1: G[i,i] = np.e ** (-2*np.pi*1j/3)
            elif i == 2*l + 2: G[i,i] = np.e ** (2*np.pi*1j/3)
            else: G[i,i] = 1
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
        k1 = dUdt(t, U, H)
        k2 = dUdt(t + 0.5*h, U + 0.5*h*k1, H)

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
    if isinstance(U0,torch.Tensor):
        I = tensor(np.eye(len(U)))
    else:
        I = np.eye(len(U))

    for i in range(n):
        if isinstance(U0,torch.Tensor):
            # print("Torch tensor SRK2")
            # print(len(U))
            # print(len(-1j*H(t+0.25*h)))
            # print(len(torch.matmul(-1j*H(t+0.25*h),U)))
            k1 = torch.linalg.solve(I + 1j*h*0.25*H(t + 0.25*h), -1j*H(t+0.25*h) @ U)
            k2 = torch.linalg.solve(I + 1j*h*0.25*H(t + 0.75*h), -1j*H(t+ 0.75*h) @ U - 1j*0.5*h*H(t+0.75*h) @ k1)
        else:
            k1 = scipy.linalg.solve(I + 1j*h*0.25*H(t + 0.25*h), -1j*H(t+0.25*h) @ U)
            k2 = scipy.linalg.solve(I + 1j*h*0.25*H(t + 0.75*h), -1j*H(t+ 0.75*h) @ U - 1j*0.5*h*H(t+0.75*h) @ k1)

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
    elif couplingType == "capacitiveCoup":
        a = np.zeros((level, level))
        for i in range(len(a)-1):
            a[i,i+1] = np.sqrt(i+1)
        H0 = kron(a + a.T,a + a.T)
    elif couplingType == "capacitiveCoupUnit":
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


# # Add section for eState occupation 
# def CTL_StateOcc(M,cType,maxDStr,level,tmin,R,anharmVal,stag,plot):
#     #Params
#     N = 2
#     id = np.eye(level)
#     h = 0.005
#     '''
#     temp
    
#     '''
#     def stateProj(eLevel,maxLevel,output=False):
#         '''
#         DESC: Creates a state with indexes only on a Hilbert SubSpace. \n

#         PARAMS: 
#             - eLevel: Energy level SubSpace of interest. 
#             - maxLevel: Total energy level of the Hilbert Space. 
#             - output: Boolean for printing statement. \n

#         OUTPUT: Vector with nonzero indexes on the given subspace. \n

#         AUTHOR: Bora Basyildiz. 
#         '''
#         psi = ones(maxLevel+1)
#         psi[eLevel] = 2
#         psi = kron(psi,psi)
#         psi = (psi != 1).astype(int)
#         for i in range(len(psi)): # This is to ensure no population of higher energy states
#             if i > eLevel*(maxLevel+1) + eLevel: psi[i] = 0
#         #psi = psi/sqrt(sum(psi))
#         if output == True:
#             for i in range(maxLevel+1):
#                 for j in range(maxLevel+1):
#                     print("|" + str(i) + str(j) + ">: " + str(psi[(maxLevel+1)*i + j]))
#         return psi

#     def vecSpaceGen(arr):
#         '''
#         DESC: Generates set of basis vectors for a given subspace. \n

#         PARAMS: 
#             - arr: Vector with non-zero entries on subspace elements (output from stateProj). \n

#         OUTPUT: Basis Vectors of Hilbert Subspace. 

#         AUTHOR: Bora Basyildiz 
        
#         '''
#         vecs = []
#         for i,val in enumerate(arr):
#             if val != 0:
#                 tempArr = zeros((len(arr)))
#                 tempArr[i] = 1
#                 vecs.append(tempArr)
#         return vecs

#     def cpNP(t,coef1,coef2,phase=0):
#             c = (maxDStr/np.sqrt(2)) *( np.cos(coef1) + 1j * np.cos(coef2))
#             p = (np.exp(1j*phase*t)) #LOL USER ERROR (I AM CONFIDENT THAT THIS CAN WORK, it is just user errors leading to divergence)
#             #shape = tensor((np.sin(np.pi * t * M / tmin)) ** 2) restart pulse envelope function 
#             #return c*p*shape
#             return c*p
#     # Help function for Probabilty Calculation
#     def spaceOcc(U,state,stateVecs):
#         occVal = 0
#         for stateVec in stateVecs:
#             occVal += (abs(stateVec.conj().T @ U @ state)) ** 2
#         return occVal

#     #Coupling Hamiltonian 
#     U_Exp = 1
#     H0 = genCouplMat(cType,level)

#     #SubSpace Occupation Variables 
#     l = level - 1
#     qbArr = stateProj(1,l) + stateProj(0,l)
#     qbStates = vecSpaceGen(qbArr)
#     qbOccVals = []

#     qtArr = stateProj(2,l)
#     qtStates = vecSpaceGen(qtArr)
#     qtOccVals = []

#     qttArr = stateProj(3,l)
#     qttStates = vecSpaceGen(qttArr)
#     qttOccVals = []


#     #Generating Unitary
#     U_Exp = 1
#     for i in range(0,N):
#         U_Exp = np.kron(U_Exp,id) #initializing unitary
    
#     qbOccup = 0
#     qtOccup = 0 
#     qttOccup = 0
#     for state in qbStates: 
#         qbOccup += spaceOcc(U_Exp,state,qbStates)
#         qtOccup += spaceOcc(U_Exp,state,qtStates)
#         qttOccup += spaceOcc(U_Exp,state,qttStates)
#     qbOccVals.append(qbOccup/len(qbStates))
#     qtOccVals.append(qtOccup/len(qtStates))
#     qttOccVals.append(qttOccup/len(qttStates))

#     for m in range(M):
#         pc = R[m]
#         def CTL_H(t):
#             HD = np.zeros((4,4),dtype='complex')
#             HD[2,2] = anharmVal
#             HD[3,3] = 2*anharmVal
#             H1 = np.copy(HD,order='K')
#             H2 = np.copy(HD,order='K')

#             #shape = tensor((np.sin(np.pi * t * M / tmin)) ** 2)
#             D1 = cpNP(t,pc[0],pc[4]) + cpNP(t,pc[1],pc[5],anharmVal) + cpNP(t,pc[2],pc[6],stag) + cpNP(t,pc[3],pc[7],stag + anharmVal)
#             D2 = cpNP(t,pc[1],pc[5]) + cpNP(t,pc[3],pc[7],anharmVal) + cpNP(t,pc[0],pc[4],-1*stag) + cpNP(t,pc[2],pc[6],-1*stag + anharmVal)

#             for i in range(len(HD)-1):
#                 H1[i,i+1] = D1
#                 H1[i+1,i] = D1.conj()

#                 H2[i,i+1] = D2
#                 H2[i+1,i] = D2.conj()
#             return H0 + np.kron(H1,id) + np.kron(id,H2)
#         U_Exp = SRK2(m/M*tmin,(m+1)/M*tmin,U_Exp,h,CTL_H)
    
#         qbOccup = 0
#         qtOccup = 0 
#         qttOccup = 0
#         for state in qbStates: 
#             qbOccup += spaceOcc(U_Exp,state,qbStates)
#             qtOccup += spaceOcc(U_Exp,state,qtStates)
#             qttOccup += spaceOcc(U_Exp,state,qttStates)
#         qbOccVals.append(qbOccup/len(qbStates))
#         qtOccVals.append(qtOccup/len(qbStates))
#         qttOccVals.append(qttOccup/len(qbStates))

#     if plot == True:
#         plt.plot(qbOccVals)
#         plt.plot(qtOccVals)
#         plt.plot(qttOccVals)
#         plt.xlabel('$M$',fontsize=16)
#         plt.ylabel('$P$',fontsize=16)
#         plt.grid(which='major', linestyle='-', linewidth='0.5')
#         plt.grid(which='minor', linestyle='dotted', linewidth='0.5')
#         plt.minorticks_on()
#         plt.legend(["QS","QT","QTT"])
#     return [qbOccVals,qtOccVals,qttOccVals]

def stateProj(eLevel,maxLevel,output=False):
    '''
    DESC: Creates a state with indexes only on a Hilbert SubSpace. \n

    PARAMS: 
        - eLevel: Energy level SubSpace of interest. 
        - maxLevel: Total energy level of the Hilbert Space. 
        - output: Boolean for printing statement. \n

    OUTPUT: Vector with nonzero indexes on the given subspace. \n

    AUTHOR: Bora Basyildiz. 
    '''
    psi = torch.ones(maxLevel+1)
    psi[eLevel] = 2
    psi = torch.kron(psi,psi)
    psi = (psi != 1).int()
    #print(psi)
    for i in range(len(psi)): # This is to ensure no population of higher energy states
        if i > eLevel*(maxLevel+1) + eLevel: psi[i] = 0
    #psi = psi/sqrt(sum(psi))
    if output == True:
        for i in range(maxLevel+1):
            for j in range(maxLevel+1):
                print("|" + str(i) + str(j) + ">: " + str(psi[(maxLevel+1)*i + j]))
    return psi

def vecSpaceGen(arr):
    '''
    DESC: Generates set of basis vectors for a given subspace. \n

    PARAMS: 
        - arr: Vector with non-zero entries on subspace elements (output from stateProj). \n

    OUTPUT: Basis Vectors of Hilbert Subspace. 

    AUTHOR: Bora Basyildiz 
    
    '''
    dt = torch.cdouble
    vecs = []
    for i,val in enumerate(arr):
        if val != 0:
            tempArr = zeros((len(arr)))
            tempArr[i] = 1
            vecs.append(tempArr)
    return torch.tensor(vecs,dtype=dt)

# Help function for Probabilty Calculation
def spaceOcc(U,state,stateVecs):
    occVal = 0
    for stateVec in stateVecs:
        occVal += (abs(stateVec.conj().T @ U @ state)) ** 2
    return occVal

# helper function to calculate maximum value of a numpy array of torch tensors 
def torchMax(list):
    maxTorchVal = list[0]
    for val in list:
        if val > maxTorchVal:
            maxTorchVal = val
    return maxTorchVal

def HC(t,Evec,l):
    n = l + 1 #number of energy levels, l is the highest energy level
    a1 = []
    a2 = []
    Evec1 = Evec[0] # split Evec into two components, then have cross components 
    Evec2 = Evec[1]
    Evec3 = Evec[2]
    for i in range(1,n):
        tempArr1 = []
        tempArr2 = []
        for j in range(i): tempArr1.append(0)
        for j in range(i): tempArr2.append(0)

        tempArr1.append(np.sqrt(i)*np.exp(-1j*(Evec1[i-1] - Evec1[i])*t))
        tempArr2.append(np.sqrt(i)*np.exp(-1j*(Evec2[i-1] - Evec2[i])*t))
        for j in range(n-1-i):
            tempArr1.append(0)
            tempArr2.append(0)
        a1.append(tempArr1)
        a2.append(tempArr2)
    a1.append([0 for i in range(n)])
    a2.append([0 for i in range(n)])

    a1 = np.array(a1)
    a2 = np.array(a2)

    resonant = 2*(np.cos((0-Evec3[0])*t) + np.cos((Evec2[2]-Evec3[0])*t) + np.cos((Evec1[2]-Evec3[0])*t) + np.cos((Evec3[1]-Evec3[0])*t))
    return resonant*(np.kron(a1 + np.conj(a1.T),a2 + np.conj(a2.T)))

# def HC(t,Evec,l):#continuous coupling Hamiltonian (stinky code, fix)
#         #g = 0.005 # coupling strength in terms of GHz
#         E_00 = Evec[0] #verbose way of defing this. Meant to illustrate what the energy levels are in Evec. 
#         E_10 = Evec[1]
#         E_01 = Evec[2]
#         E_11 = Evec[3]
#         E_02 = Evec[4]
#         E_20 = Evec[5]
#         E_22 = Evec[6]
#         E_30 = Evec[7]
#         E_03 = Evec[8]
#         # E_00 = 0 #experimentally defined values 
#         # E_10 = 4.994/g
#         # E_01 = 5.440/g
#         # E_11 = 10.433/g
#         # E_02 = 10.681/g
#         # E_20 = 9.832/g
#         # E_22 = 20.506/g

#         omega1 = E_00 - E_11
#         omega2 = E_02 - E_11
#         omega3 = E_20 - E_11
#         omega4 = E_22 - E_11

#         if l == 3:
#             #second quantization operators 
#             a1 = np.array([[0,np.exp(-1j*(E_00 - E_10)*t),0],[0,0,np.sqrt(2)*np.exp(-1j*(E_10 - E_20)*t)],[0,0,0]])
#             a2 = np.array([[0,np.exp(-1j*(E_00 - E_01)*t),0],[0,0,np.sqrt(2)*np.exp(-1j*(E_01 - E_02)*t)],[0,0,0]])
#             resonantP = 2*(np.cos(omega1*t) + np.cos(omega2*t) + np.cos(omega3*t) + np.cos(omega4*t))
#             H0 = np.kron(a1 + np.conj(a1.T),a2 + np.conj(a2.T))
#             # Hreturn = resonantP*H0
#             # if isinstance(Evec,torch.Tensor):
#             #     Hreturn = torch.tensor(Hreturn)
#             # return Hreturn
#             return resonantP*H0
#         elif l ==4:
#             a1 = np.array([[0,np.exp(-1j*(E_00 - E_10)*t),0,0],[0,0,np.sqrt(2)*np.exp(-1j*(E_10 - E_20)*t),0],[0,0,0,np.sqrt(3)*np.exp(-1j*(E_20 - E_30)*t)],[0,0,0,0]])
#             a2 = np.array([[0,np.exp(-1j*(E_00 - E_01)*t),0,0],[0,0,np.sqrt(2)*np.exp(-1j*(E_01 - E_02)*t),0],[0,0,0,np.sqrt(3)*np.exp(-1j*(E_02 - E_03)*t)],[0,0,0,0]])
#             resonantP = 2*(np.cos(omega1*t) + np.cos(omega2*t) + np.cos(omega3*t) + np.cos(omega4*t))
#             H0 = np.kron(a1 + np.conj(a1.T),a2 + np.conj(a2.T))
#             return resonantP*H0
#         else:
#             n = l + 1 #number of energy levels, l is the highest energy level
#             Evec = []
#             a1 = []
#             a2 = []
#             Evec1 = np.zeros(n) # split Evec into two components, then have cross components 
#             Evec2 = np.zeros(n)
#             for i in range(1,n):
#                 tempArr1 = []
#                 tempArr2 = []
#                 for j in range(i): tempArr1.append(0)
#                 for j in range(i): tempArr2.append(0)

#                 tempArr1.append(np.sqrt(i)*np.exp(-1j*(Evec1[i-1] - Evec1[i])*t))
#                 tempArr2.append(np.sqrt(i)*np.exp(-1j*(Evec2[i-1] - Evec2[i])*t))
#                 for j in range(n-1-i):
#                     tempArr1.append(0)
#                     tempArr2.append(0)
#                 a1.append(tempArr1)
#                 a2.append(tempArr2)
#             a1.append([0 for i in range(n)])
#             a2.append([0 for i in range(n)])

#             a1 = np.array(a1)
#             a2 = np.array(a2)

#             resonant = 2*(np.cos((Evec[0]-Evec[2*n+1])*t) + np.cos((Evec[2]-Evec[2*n+1])*t) + np.cos((Evec[n+2]-Evec[2*n+1])*t) + np.cos((Evec[2*n+2]-Evec[2*n+1])*t))
#             return resonant*(np.kron(a1 + np.conj(a1.T),a2 + np.conj(a2.T)))

# def genEvec(anharm1,anharm2,stag,g): #Turn into generator for arbitrary energy levels (stinky code)
#         E10 = 500/3 #Set qubit intial frequency to 5 GHz. Here we assume that g ~ 30 MHz.
#         E01 = E10 + stag*g

#         E11 = E10 + E01
#         E20 = 2*E10 + anharm1*g
#         E02 = 2*E01 + anharm2*g
#         #E12 = E10 + E02
#         #E21 = E20 + E01
#         E22 = E20 + E02
#         E30 = 3*E10 + 2*anharm1*g
#         E03 = 3*E01 + 2*anharm2*g
#         return np.array([0,E01,E02,])

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
    E01 = E10 + stag*g

    Evec1 = [0,E10] #Energy levels for qubit 1
    for i in range(1,level+1):
        Evec1.append((i+1)*E10+i*anharm1*g)

    Evec2 = [0,E01] #Energy levels for qubit 2
    for i in range(1,level+1):
        Evec2.append((i+1)*E01+i*anharm2*g)

    E11 = E10 + E01
    E22 = Evec1[2] + Evec2[2]
    Evec3 = [E11,E22] # Energy levels for multi-excited states

    return np.array([Evec1,Evec2,Evec3])


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


def genTwoQuditBasis(d,l,dt):#need to be changed to d and level when extending to leakage systems
        SU = []
        N = 2

        generators = genQuditBasis(d,l)

        perms = list(product(range(len(generators)),repeat=N))#all permutations of orthonormal basis of two qudits

        idTemp = np.zeros((l,l))
        for i in range(d): idTemp[i,i] = 1

        #Making Pauli Basis
        for tuple in perms:
            gen1 = generators[tuple[0]]
            gen2 = generators[tuple[1]]
            
            if tuple[0] == 0:
                 gen1 = idTemp
            if tuple[1] == 0:
                 gen2 = idTemp

            tempU = torch.tensor(np.kron(gen1,gen2),dtype=dt)
            SU.append(tempU)
        return SU
