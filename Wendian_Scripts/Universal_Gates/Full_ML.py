#Imports are here too to export block to Wendian script
from numpy import array,zeros,kron
from torch import matmul,trace,matrix_exp,tensor,manual_seed
import torch
from itertools import product
import numpy as np

def fidelity_ml(M,input_gate,t,N_iter,rseed,H0,drives):
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Mon Aug 16 4:33 2021

    @author: Bora & Alex


    DESC: This function does both qubit and qutrit optimizations for a given random seed and 
          coupling matrix. 
    """
    #Variable Initializations
    N = 2 #This code is only working for 2 qudits. Adapt to N qudits in the future
    level = len(drives[0])
    manual_seed(rseed)
    dt = torch.cdouble
    infidelity_list=torch.zeros([N_iter,1])
    id = np.eye(level)
    H0 = tensor(H0,dtype=dt) #if H0 is numpy array, convert to torch tensor 
    input_gate = tensor(input_gate,dtype=dt)
    Diagonal = False


    #Diagonal Entry calculation
    if torch.all(H0 == 0):
        dentries = tensor(np.random.rand(5),dtype=dt)*2*np.pi
        dentries.requires_grad = True
        temp_vec = tensor([0,0,1,0])
        catvec = torch.cat((temp_vec, torch.cos(dentries)), dim=0)
        H0 = np.zeros([3 ** 2, 3 ** 2])
        H0 = torch.tensor(H0)
        H0[2,2] = 1
        for i in range(5):
            H0[3 ** 2 - i-1,3 ** 2 - i-1] = 1
        H0 = torch.mul(H0,catvec)
        Diagonal = True


    #Sums Pauli gates with coefficients 
    def sum_pauli(coef, gate):
        total_pauli = torch.tensor(zeros([level ** N, level ** N]))
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
    
    def gen_SU():
        SU = []
        pauli_int = [1,2,3,4]
        perms = list(product(pauli_int,repeat=N))#all permutations of paulis

        #Paul Matrices only in the qubit space 
        if level == 2:
            sx = array([[0, 1], [1, 0]])
            sy = array([[0,-1j],[1j,0]])
            sz = array([[1, 0], [0, -1]])
            id = array([[1,0],[0,1]])
        elif level == 3 :
            sx = array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) 
            sy = array([[0,-1j, 0],[1j,0, 0], [0, 0, 0]]) 
            sz = array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) 
            id = array([[1, 0, 0],[0, 1, 0],[0, 0, 0]]) #note that id has to redefined as it is non-unitary in the SU(2) subgroup of qutrits. This will not override the parent id variable.
        else: raise Exception("Incorrect qudit level! (either 2 or 3)")

        #Making Pauli Basis
        for p in perms:
                unitary = 1
                for pauli in p:
                    if pauli == 1: unitary = tensor(kron(unitary,sx),dtype=dt)
                    elif pauli == 2: unitary = tensor(kron(unitary,sy),dtype=dt)
                    elif pauli == 3: unitary = tensor(kron(unitary,sz),dtype=dt)
                    elif pauli == 4: unitary = tensor(kron(unitary,id),dtype=dt)
                SU.append(unitary)
        return SU

    #Error checking
    if np.shape(input_gate.detach().numpy()) != (level ** N, level ** N):
        raise Exception("Incorret Target Gate size")
    if np.shape(H0.detach().numpy()) != (level ** N, level ** N):
        raise Exception("Incorret Coupling Matrix size")
    for d in drives:
        if np.shape(d)[0] != np.shape(d)[1]: raise Exception("Incorrect drive dimensions. Must be square.")
        if np.shape(d)[0] != level: raise Exception("All drives must the qudit size.")
    

    #PyTorch Parameter Optimization 
    R = torch.rand([M,len(drives)*N], dtype=torch.double) *2*np.pi
    R.requires_grad = True # set flag so we can backpropagate
    optimizer = torch.optim.SGD([R], lr = 0.3, momentum=0.99, nesterov=True)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.03, factor=0.3, patience= 20)

    # Generating time evolution and optimizing 
    for n in range(0,N_iter):
        #Creating Drive Hamilontian
        U_Exp = 1
        for i in range(0,N):
            U_Exp = tensor(kron(U_Exp,id),dtype=dt)#initializing unitary
        for m in range(0,M):#Product of pulses
            pulse_coef = R[m]
            H1 = 0
            for i,d in enumerate(drives):
                H1 = H1 + sum_pauli(pulse_coef[i*N:(i+1)*N],d)
            U_Exp = matmul(matrix_exp(-1j*(H0+H1)*t/M),U_Exp)

        #Fidelity calulcation given by Nielsen Paper
        fidelity = 0
        d = 2**N
        SU = gen_SU()

        for U in SU:
            eps_U = matmul(matmul(U_Exp,U),(U_Exp.conj().T))
            target_U = matmul(matmul(input_gate,(U.conj().T)),(input_gate.conj().T))
            tr = trace(matmul(target_U,eps_U))
            fidelity = fidelity + tr 
        fidelity = abs(fidelity + d*d)/(d*d*(d+1))    
        infidelity = 1 - fidelity
        infidelity_list[n] = infidelity.detach()
        infidelity.backward(retain_graph=True)

        #Printing statement
        #if (n+1)%100==0: print('Itertation ', str(n+1), ' out of ', str(N_iter), 'complete. Avg Infidelity: ', str(infidelity.item()))

        #optimizer 
        optimizer.step()
        scheduler.step(infidelity)
        optimizer.zero_grad()

        if 1 - infidelity_list[n] >= 99.99: #Stopping condition for high fidelity iterations
            return infidelity_list.min().item()
    if not Diagonal: return [1 - infidelity_list.min().item(),R.detach().numpy()]
    else: return [1 - infidelity_list.min().item(),R.detach().numpy(),dentries.detach().numpy()]