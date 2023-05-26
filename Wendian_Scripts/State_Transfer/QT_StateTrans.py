#Imports are here too to export block to Wendian script
from torch import tensor
import torch
import numpy as np

def stateTransfer_ml(M,t,N_iter,rseed,H0):
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Mon Aug 16 4:33 2021

    @author: Bora & Alex


    DESC: This function deos the state transfer protocol for a qutrit system. 
    """


    #Pauli Matricies in Qutrit Space
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
    # n3 = np.array([[0,0,0],[0,1,0],[0,0,2]]) #qubit number operator
    # H0 = torch.tensor(np.kron(n3,n3))

    #These are the coefficients we are optimizing
    R = torch.rand([M,4*N], dtype=torch.double) *2*np.pi # Random initialization (between 0 and 2pi)
    R.requires_grad = True # set flag so we can backpropagate

    optimizer = torch.optim.SGD([R], lr = 0.3, momentum=0.99, nesterov=True)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.03, factor=0.3 , patience= 20 )

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
        zero_state = np.array([1,0,0])
        one_state = np.array([0,1,0])
        OO_state = np.kron(zero_state,zero_state)
        ll_state = np.kron(one_state,one_state)
        initial_state = tensor(OO_state,dtype=dt)
        target_state = tensor(1/np.sqrt(2) * (OO_state + ll_state),dtype=dt)

        fidelity = abs(torch.dot(U_Exp @ initial_state,target_state)) ** 2
        
        infidelity = 1 - fidelity
        infidelity_list[n] = infidelity.detach()
        infidelity.backward()

        #Printing statement
        #if (n+1)%100==0: print('Itertation ', str(n+1), ' out of ', str(N_iter), 'complete. Avg Infidelity: ', str(infidelity.item()))

        #optimizer 
        optimizer.step()
        scheduler.step(infidelity)
        optimizer.zero_grad()

        if fidelity - 0.9999 > 0: #Stopping condition for high fidelity iterations
            return [1 - infidelity_list.min().item(),R.detach().numpy()]

    return [1 - infidelity_list.min().item(),R.detach().numpy()]