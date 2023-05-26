#Imports are here too to export block to Wendian script
import torch
import numpy as np
from itertools import product

def fidelity_ml(M,input_gate,t,N_iter,g,rseed):
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Mon Aug 16 4:33 2021

    @author: Bora & Alex


    DESC: This function tests the difference in coupling strength for qutrit drives. 
    """


    #Pauli Matricies in Qubit Subspace
    sx = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) 
    sy = np.array([[0,-1j, 0],[1j,0, 0], [0, 0, 0]]) 
    sxx = np.array([[0,0,0],[0,0,1],[0,1,0]]) 
    syy = np.array([[0,0,0],[0,0,-1j],[0,1j,0]]) 
    id = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]) 

    
    #Function definitions 
    
    def sum_pauli(coef, gate):#Sums Pauli gates with coefficients 
        total_pauli = torch.tensor(np.zeros([3 ** N, 3 ** N]))
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

    #variable initializations
    N = 2
    torch.manual_seed(rseed)
    dt = torch.cdouble # datatype and precision
    infidelity_list=torch.zeros([N_iter,1])

    #Coupling terms in Hamiltonian 
    n3 = np.array([[0,0,0],[0,1,0],[0,0,2]]) #qubit number operator
    H0 = torch.tensor(np.kron(n3,n3))

    #These are the coefficients we are optimizing
    #R = torch.rand([M,2*N], dtype=torch.double) *2*np.pi # Random initialization (between 0 and 2pi)
    R = torch.rand([M,4*N], dtype=torch.double) *2*np.pi # Random initialization (between 0 and 2pi)
    R.requires_grad = True # set flag so we can backpropagate
    optimizer = torch.optim.SGD([R], lr = 0.3, momentum=0.99, nesterov=True)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.03, factor=0.3 , patience= 20 )


    #Unitary group generation
    SU = []
    pauli_int = [1,2,3,4]
    perms = list(product(pauli_int,repeat=N))#all permutations of paulis
    #Paul Matrices only in the qubit space 
    sxq = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) 
    syq = np.array([[0,-1j, 0],[1j,0, 0], [0, 0, 0]]) 
    szq = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) 
    idq = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 0]]) 
    for p in perms:#mapping integers to pauli 
        unitary = 1
        for pauli in p:
            if pauli == 1:
                unitary = torch.tensor(np.kron(unitary,sxq),dtype=torch.cdouble)
            elif pauli == 2:
                unitary = torch.tensor(np.kron(unitary,syq),dtype=torch.cdouble)
            elif pauli == 3:
                unitary = torch.tensor(np.kron(unitary,szq),dtype=torch.cdouble)
            elif pauli == 4:
                unitary = torch.tensor(np.kron(unitary,idq),dtype=torch.cdouble)
        SU.append(unitary)

    #prev_infidelity = -1
    for n in range(0,N_iter):
        #Creating Drive Hamilontian
        U_Exp = 1
        for i in range(0,N):
            U_Exp = torch.tensor(np.kron(U_Exp,id),dtype=dt)#initializing unitary
        for m in range(0,M):#Product of pulses
            pulse_coef = R[m]
            H1 = sum_pauli(pulse_coef[:N],sx) + sum_pauli(pulse_coef[N:2*N],sy) + sum_pauli(pulse_coef[2*N:3*N],sxx) + sum_pauli(pulse_coef[3*N:4*N],syy) 
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