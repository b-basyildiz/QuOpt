from numpy import array,zeros,kron,outer
from torch import matmul,trace,matrix_exp,tensor
import torch
from itertools import product
import numpy as np
def fidelity_ml(M,target_gate,t,N_iter,g,rseed):
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Mon Aug 16 4:33 2021

    @author: Bora & Alex

    This is the original code for Joel's Paper.

    Note that we have XX coupling in this script at the moment
    """
    
    #Pauli Matricies
    sx = array([[0, 1], [1, 0]])
    sy = array([[0,-1j],[1j,0]])
    sz = array([[1, 0], [0, -1]])
    id = array([[1,0],[0,1]])
    
    def sum_pauli(coef, gate):#Sums Pauli gates with coefficients 
        N = len(coef)#number of qubits
        total_pauli = tensor(zeros([2 ** N,2 ** N]))
        #Summing all Z gates
        for i in range(0,N):
            pauli_temp = 1
            for j in range(0,i):
                pauli_temp = tensor(kron(pauli_temp,id))
            pauli_temp = tensor(kron(pauli_temp,gate))
            for j in range(i+1,N):
                pauli_temp = tensor(kron(pauli_temp,id))
            total_pauli = total_pauli + coef[i]*pauli_temp
        return total_pauli

    #variable initializations
    N = 2
    torch.manual_seed(rseed)
    dt = torch.cdouble # datatype and precision
    infidelity_list=torch.zeros([N_iter,1])

    #H0 generation
    # def qubitSubspace(gate):
    #     gate = np.delete(gate,2,0)
    #     gate = np.delete(gate,2,1)
    #     return gate[:4,:4]
    # H = np.zeros([3 ** 2, 3 ** 2])
    # H[1,3] = 1
    # H[2,3] = 1
    # H0 = H + H.transpose()
    # H0 = torch.tensor(g*qubitSubspace(H0))
    H0 = tensor(g*kron(sx,sx))


    #Unitary group generation
    SU = []
    pauli_int = [1, 2, 3, 4]#eq to [sx,sy,sz,id]
    perms = list(product(pauli_int,repeat=N))#all permutations of paulis
    for p in perms:#mapping integers to pauli 
        unitary = 1
        for pauli in p:
            if pauli == 1: unitary = tensor(kron(unitary,sx),dtype=dt)
            elif pauli == 2: unitary = tensor(kron(unitary,sy),dtype=dt)
            elif pauli == 3: unitary = tensor(kron(unitary,sz),dtype=dt)
            elif pauli == 4: unitary = tensor(kron(unitary,id),dtype=dt)
        SU.append(unitary)

    #These are the coefficients we are optimizing
    R = torch.rand([M,2*N], dtype=torch.double) *2*np.pi # Random initialization (between 0 and 2pi)
    R.requires_grad = True # set flag so we can backpropagate

    #Optimizer settings(can be changed & opttimized)
    lr=0.3#learning rate
    optimizer = torch.optim.SGD([R], lr = lr, momentum=0.99, nesterov=True)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.03, factor=0.3 , patience= 20)

    for n in range(0,N_iter):
        #Creating Hamilontian
        U_Exp = 1
        for i in range(0,N):
            U_Exp = tensor(kron(U_Exp,id),dtype=dt)#initializing unitary
        for m in range(0,M):#Product of pulses
            pulse_coef = R[m]
            H1 = sum_pauli(pulse_coef[:N],sx) + sum_pauli(pulse_coef[N:],sy)
            U_Exp = matmul(matrix_exp(-1j*(H0+H1)*t/M),U_Exp)

        #Fidelity calulcation given by Nielsen Paper
        fidelity = 0
        d = 2**N
        for U in SU:
            eps_U = matmul(matmul(U_Exp,U),(U_Exp.conj().T))
            target_U = matmul(matmul(target_gate,(U.conj().T)),(target_gate.conj().T)) #Double Check this Calculation, why is U conjugated at this point? artifact from inner product 
            tr = trace(matmul(target_U,eps_U))
            fidelity = fidelity + tr
        fidelity = abs(fidelity + d*d)/(d*d*(d+1))    
        infidelity = 1 - fidelity
        infidelity_list[n] = infidelity.detach()
        infidelity.backward()

        #Printing statement
        #if (n+1)%20==0: print('Itertation ', str(n+1), ' out of ', str(N_iter), 'complete. Avg Infidelity: ', str(infidelity.item()))

        #optimizer 
        optimizer.step()
        scheduler.step(infidelity)
        optimizer.zero_grad()
        
        if 1 - infidelity_list[n] >= 99.99: #Stopping condition for high fidelity iterations
            return infidelity_list.min().item()
    
    return [1-infidelity_list.min().item(),R.detach().numpy()]