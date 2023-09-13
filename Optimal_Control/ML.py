#Imports are here too to export block to Wendian script
from numpy import array,zeros,kron,exp,diag,concatenate
from torch import matmul,trace,matrix_exp,tensor,manual_seed,flip
import torch
from itertools import product
import numpy as np
from helperFuncs import *

def fidelity_ml(M,input_gate,t,N_iter,rseed,H0,drives,maxDriveStrength,leakage,crossTalk,h,stag):
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Mon Aug 16 4:33 2021

    @author: Bora & Alex


    DESC: This function does both qubit and qutrit optimizations for a given random seed and 
          coupling matrix. 
    """

    #Modeling Type
    ctBool= True
    if crossTalk == "False": ctBool = False
    if leakage: #for modeling leakage
        quditDrives = drives[1]
        anharm = drives[2]
        drives = drives[0]
        anharmVal = float(anharm[-1,-1])

    #Variable Initializations 
    N = 2 #This code is only working for 2 qudits. Adapt to N qudits in the future
    level = len(drives[0])
    #rseed = 1 #fixed for cross talk testing
    manual_seed(rseed)
    #torch.set_default_dtype(torch.cdouble)
    dt = torch.cdouble
    infidelity_list=torch.zeros([N_iter,1])
    id = np.eye(level)
    H0 = tensor(H0,dtype=dt) #if H0 is numpy array, convert to torch tensor 
    input_gate = tensor(input_gate,dtype=dt)


    #Sums Pauli gates with coefficients 
    def sum_pauli(coef, gate,tm=0):
        total_pauli = torch.tensor(zeros([level ** N, level ** N]))
        for i in range(0,N):
            pauli_temp = 1
            for j in range(0,i):
                pauli_temp = torch.tensor(np.kron(pauli_temp,id))
            pauli_temp = torch.tensor(np.kron(pauli_temp,gate))
            for j in range(i+1,N):
                pauli_temp = torch.tensor(np.kron(pauli_temp,id))
            phase = tensor(exp((-1) ** (i+1) * 1j*stag*tm))  #time dependent phase, only non-zero for cross talk modeling 
            if maxDriveStrength == -1: total_pauli = total_pauli + phase*coef[i]*pauli_temp #if maxDriveStrength = -1, then we have unlimited drive strength
            else: total_pauli = total_pauli + phase*maxDriveStrength*torch.cos(coef[i])*pauli_temp
        return total_pauli
    
    def gen_SU():
        SU = []
        pauli_int = [1,2,3,4]
        perms = list(product(pauli_int,repeat=N))#all permutations of paulis

        sxqb = genDrive(level,1,"x")
        syqb = genDrive(level,1,"y")
        idqb = zeros((level, level))
        idqb[0,0] = 1  
        idqb[1,1] = 1
        szqb = diag(concatenate((array([1,-1]),array((level-2)*[0]))))

        #Making Pauli Basis
        for p in perms:
                unitary = 1
                for pauli in p:
                    if pauli == 1: unitary = tensor(kron(unitary,sxqb),dtype=dt)
                    elif pauli == 2: unitary = tensor(kron(unitary,syqb),dtype=dt)
                    elif pauli == 3: unitary = tensor(kron(unitary,idqb),dtype=dt)
                    elif pauli == 4: unitary = tensor(kron(unitary,szqb),dtype=dt)
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
        segCt = 0
        for i in range(0,N):
            U_Exp = tensor(kron(U_Exp,id),dtype=dt)#initializing unitary

        for m in range(0,M):#Product of pulses
            pulse_coef = R[m]
            H1 = 0
            for i,d in enumerate(drives):
                H1 = H1 + sum_pauli(pulse_coef[i*N:(i+1)*N],d)
                if leakage:  H1 = H1 + sum_pauli(pulse_coef[i*N:(i+1)*N],quditDrives[(i % len(quditDrives))]) 
            if leakage: H1 = H1 + sum_pauli(tensor([1]*N),anharm)
            H = H0 + H1
            if ctBool:
                def Ht(tin):
                    H1t = torch.zeros((len(H),len(H)))
                    for i,d in enumerate(drives):
                        H1t = H1t + sum_pauli(flip(pulse_coef[i*N:(i+1)*N],dims=[0]),d,tin)  #Cross talk drives with time dependet phases and fliped coefficients. 
                    H1t = H1t + H1t.conj().T # Hermitian Conjugate
                    return H + H1t
                #htemp = h*t/M
                if crossTalk == "RK2": U_Exp = normU(RK2(m/M*t,(m+1)/M*t,U_Exp,h,dUdt,Ht))
                elif crossTalk == "SV2": U_Exp = SV2(m/M*t,(m+1)/M*t,U_Exp,h,Ht)
                elif crossTalk == "SRK2": U_Exp = SRK2(m/M*t,(m+1)/M*t,U_Exp,h,Ht)
                else: raise Exception("Incorrect Cross Talk Modeling Type. Either Second Order Runge-Kutta, Störmer-Verlet, or symplectic Runge-Kutta.")
                #U_ExpCT = normU(U_ExpCT)
                #U_Exp = (matmul(U_ExpCT,U_Exp)) # Matrix evolution
            else: 
                #normPrint(U_Exp)
                U_Exp = matmul(matrix_exp(-1j*(H)*t/M),U_Exp)
                #normPrint((U_Exp.detach().numpy()))
            #print(U_Exp)

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
        #if (n+1)%1==0: print('Itertation ', str(n+1), ' out of ', str(N_iter), 'complete. Avg Infidelity: ', str(infidelity.item()))

        #optimizer 
        optimizer.step()
        scheduler.step(infidelity)
        optimizer.zero_grad()

        # if fidelity.detach() >= 0.9999:
        #     print("made it to the end 1")
        #     return [1 - infidelity_list.min().item(),R.detach().numpy()]
    # for row in U_Exp.detach().numpy():
    #     for val in row:
    #         print(round(abs(val),1),end=" ")
    #     print()
    return [1 - infidelity_list.min().item(),R.detach().numpy()]