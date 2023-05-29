#Imports are here too to export block to Wendian script
from numpy import array,zeros,kron,outer
from torch import matmul,trace,matrix_exp,tensor
import torch
from itertools import product
import numpy as np

def fidelity_ml(M,input_gate,t,N_iter,rseed,H0,level,subspace):
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Mon Aug 16 4:33 2021

    @author: Bora & Alex


    DESC: This function does both qubit and qutrit optimizations for a given random seed and 
          coupling matrix. 
    """
    #Sums Pauli gates with coefficients 
    def sum_pauli(coef, gate):
        total_pauli = torch.tensor(np.zeros([level ** N, level ** N]))
        for i in range(0,N):
            pauli_temp = 1
            for j in range(0,i):
                pauli_temp = torch.tensor(np.kron(pauli_temp,id))
            pauli_temp = torch.tensor(np.kron(pauli_temp,gate))
            for j in range(i+1,N):
                pauli_temp = torch.tensor(np.kron(pauli_temp,id))
            #total_pauli = total_pauli + maxFreq*torch.cos(coef[i])*pauli_temp
            total_pauli = total_pauli + coef[i]*pauli_temp
        return total_pauli

    #Pauli Matrices in either space 
    drives = []
    if level == 2: #Qubit Subspace
        sx = array([[0, 1], [1, 0]])
        sy = array([[0,-1j],[1j,0]])
        sz = array([[1, 0], [0, -1]])
        id = array([[1,0],[0,1]])
        drives.append(sx)
        drives.append(sy)
    elif level == 3: #Qutrit Subspace
        sx = array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) 
        sy = array([[0,-1j, 0],[1j,0, 0], [0, 0, 0]]) 
        sxx = array([[0,0,0],[0,0,1],[0,1,0]]) 
        syy = array([[0,0,0],[0,0,-1j],[0,1j,0]]) 
        id = array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]) 
        if subspace == 0:
            drives.append(sx)
            drives.append(sy)
            drives.append(sxx)
            drives.append(syy)
        elif subspace == 1:
            drives.append(sxx)
            drives.append(syy)
        elif subspace == 2:
            drives.append(syy)
        else: raise Exception("Incorrect subspace input (0,1,2)")
    else:
        raise Exception("Incorrect qudit level! (either 2 or 3)")

    #variable initializations
    N = 2
    torch.manual_seed(rseed)
    dt = torch.cdouble # datatype and precision
    infidelity_list=torch.zeros([N_iter,1])

    #Error checking

    if np.shape(input_gate.detach().numpy()) != (level ** N, level ** N):
        raise Exception("Incorret Target Gate size")
    if np.shape(H0.detach().numpy()) != (level ** N, level ** N):
        raise Exception("Incorret Coupling Matrix size")

    #These are the coefficients we are optimizing (Random initialization between 0 and 2pi)
    R = torch.rand([M,len(drives)*N], dtype=torch.double) *2*np.pi
    R.requires_grad = True # set flag so we can backpropagate
    optimizer = torch.optim.SGD([R], lr = 0.3, momentum=0.99, nesterov=True)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.03, factor=0.3 , patience= 20 )


    #Unitary group generation
    SU = []
    pauli_int = [1,2,3,4]
    perms = list(product(pauli_int,repeat=N))#all permutations of paulis
    #Paul Matrices only in the qubit space 
    if level == 2:
        for p in perms:#mapping integers to pauli 
            unitary = 1
            for pauli in p:
                if pauli == 1: unitary = tensor(kron(unitary,sx),dtype=dt)
                elif pauli == 2: unitary = tensor(kron(unitary,sy),dtype=dt)
                elif pauli == 3: unitary = tensor(kron(unitary,sz),dtype=dt)
                elif pauli == 4: unitary = tensor(kron(unitary,id),dtype=dt)
            SU.append(unitary)
    elif level == 3 :
        sxq = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) 
        syq = np.array([[0,-1j, 0],[1j,0, 0], [0, 0, 0]]) 
        szq = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) 
        idq = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 0]]) 
        for p in perms:#mapping integers to pauli 
            unitary = 1
            for pauli in p:
                if pauli == 1: unitary = torch.tensor(np.kron(unitary,sxq),dtype=dt)
                elif pauli == 2:
                    unitary = torch.tensor(np.kron(unitary,syq),dtype=dt)
                elif pauli == 3:
                    unitary = torch.tensor(np.kron(unitary,szq),dtype=dt)
                elif pauli == 4:
                    unitary = torch.tensor(np.kron(unitary,idq),dtype=dt)
            SU.append(unitary)
    else: raise Exception("Incorrect qudit level! (either 2 or 3)")

    for n in range(0,N_iter):
        #Creating Drive Hamilontian
        U_Exp = 1
        for i in range(0,N):
            U_Exp = torch.tensor(np.kron(U_Exp,id),dtype=dt)#initializing unitary
        for m in range(0,M):#Product of pulses
            pulse_coef = R[m]
            H1 = 0
            for i,d in enumerate(drives):
                H1 = H1 + sum_pauli(pulse_coef[i*N:(i+1)*N],d)
            U_Exp = torch.matmul(torch.matrix_exp(-1j*(H0+H1)*t/M),U_Exp)

        #Fidelity calulcation given by Nielsen Paper
        fidelity = 0
        d = 2**N
        for U in SU:
            eps_U = torch.matmul(torch.matmul(U_Exp,U),(U_Exp.conj().T))
            target_U = torch.matmul(torch.matmul(input_gate,(U.conj().T)),(input_gate.conj().T))
            tr = torch.trace(torch.matmul(target_U,eps_U))
            fidelity = fidelity + tr 
        fidelity = abs(fidelity + d*d)/(d*d*(d+1))    
        infidelity = 1 - fidelity
        infidelity_list[n] = infidelity.detach()
        infidelity.backward()

        #Printing statement
        #if (n+1)%100==0: print('Itertation ', str(n+1), ' out of ', str(N_iter), 'complete. Avg Infidelity: ', str(infidelity.item()))

        #optimizer 
        optimizer.step()
        scheduler.step(infidelity)
        optimizer.zero_grad()

        if 1 - infidelity_list[n] >= 99.99: #Stopping condition for high fidelity iterations
            return infidelity_list.min().item()
    return [1 - infidelity_list.min().item(),R.detach().numpy()]