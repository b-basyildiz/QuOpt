#Imports are here too to export block to Wendian script
from numpy import array,zeros,kron,exp,diag,concatenate
from torch import matmul,trace,matrix_exp,tensor,manual_seed,flip
import torch
from itertools import product
import numpy as np
from helperFuncs import *
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

def fidelity_ml(M,input_gate,tmin,dspaceLen,N_iter,rseed,H0,drives,maxDriveStrength,lbool,minLeak,crossTalk,h,alpha,anharmVal,stag,ode,ContPulse,optimizer,weightFName,wsmBoolF):
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Mon Aug 16 4:33 2021

    @author: Bora 


    DESC: This function does both qubit and qutrit optimizations for a given random seed and 
          coupling matrix. 
    """

    #Modeling Type
    ctBool= True # Cross Talk
    if crossTalk == "False": ctBool = False

    ContBool = True #Continuous Pulses
    if ContPulse == "False": ContBool = False




    if lbool: #for modeling leakage
        quditDrives = drives[1]
        anharm = drives[2]
        drives = drives[0]
        anharmVal = float(anharm[-1,-1])


    CTLBool = False
    if lbool and ctBool: CTLBool = True

    #ml stands for minimize leakage.
    mlbool = False
    if minLeak == "True":
        mlbool = True

    ContHBool = False
    # Tests if coupling Hamiltonian is time dependent or independent
    try: 
        temp = H0(1)
        ContHBool = True
    except:
        pass

    warmStartBool=False
    if weightFName != "False":
        warmStartBool=True




    #Variable Initializations 
    N = 2 #This code is only working for 2 qudits. Adapt to N qudits in the future
    level = len(drives[0])
    manual_seed(rseed)
    dt = torch.cdouble
    infidelity_list=torch.zeros([N_iter,1])
    weight_list = []
    id = np.eye(level)
    if not ContHBool: H0 = tensor(H0,dtype=dt) #if H0 is numpy array, convert to torch tensor 
    input_gate = tensor(input_gate,dtype=dt)

    #Sums Pauli gates with coefficients 
    def sum_pauli(coef, gate,t=0,phaseDiff=0):
        total_pauli = torch.tensor(zeros([level ** N, level ** N]))
        for i in range(0,N):
            pauli_temp = 1
            for j in range(0,i):
                pauli_temp = torch.tensor(np.kron(pauli_temp,id))
            pauli_temp = torch.tensor(np.kron(pauli_temp,gate))
            for j in range(i+1,N):
                pauli_temp = torch.tensor(np.kron(pauli_temp,id))


            phase = tensor(exp((-1) ** (i+1) * 1j*phaseDiff*t))  #time dependent phase, only non-zero for cross talk modeling 

            # if CTLBool:
            #     total_pauli = total_pauli + coef[i]*pauli_temp
            if maxDriveStrength == -1: 
                if ContBool: total_pauli = total_pauli + phase*coef[i]* ((np.sin(np.pi * t * M / tmin)) ** 2) * pauli_temp #if maxDriveStrength = -1, then we have unlimited drive strength
                else: total_pauli = total_pauli + phase*coef[i]*pauli_temp
            else: 
                if ContBool: total_pauli = total_pauli + phase*(maxDriveStrength*torch.cos(coef[i])) * tensor((np.sin(np.pi * t * M / tmin)) ** 2) *pauli_temp
                else : total_pauli = total_pauli + phase*maxDriveStrength*torch.cos(coef[i])*pauli_temp
        return total_pauli
    
    #Continuous pulses for qutrit Cross-Talk-Leakage (CTL) modeling
    def cp(t,coef1,coef2,phase=0):
        c = tensor(maxDriveStrength/np.sqrt(2)) *( torch.cos(coef1) + 1j * torch.cos(coef2))
        p = tensor(np.exp(1j*phase*t)) 
        return c*p
    
    # def gen_SU():
    #     SU = []
    #     pauli_int = [1,2,3,4]
    #     perms = list(product(pauli_int,repeat=N))#all permutations of paulis

    #     sxqb = genDrive(level,1,"x")
    #     syqb = genDrive(level,1,"y")
    #     idqb = zeros((level, level))
    #     idqb[0,0] = 1  
    #     idqb[1,1] = 1
    #     szqb = diag(concatenate((array([1,-1]),array((level-2)*[0]))))

    #     #Making Pauli Basis
    #     for p in perms:
    #             unitary = 1
    #             for pauli in p:
    #                 if pauli == 1: unitary = tensor(kron(unitary,sxqb),dtype=dt)
    #                 elif pauli == 2: unitary = tensor(kron(unitary,syqb),dtype=dt)
    #                 elif pauli == 3: unitary = tensor(kron(unitary,idqb),dtype=dt)
    #                 elif pauli == 4: unitary = tensor(kron(unitary,szqb),dtype=dt)
    #             SU.append(unitary)
    #     return SU

    #Error checking
    if np.shape(input_gate.detach().numpy()) != (level ** N, level ** N):
        raise Exception("Incorret Target Gate size")
    if not ContHBool:
        if np.shape(H0.detach().numpy()) != (level ** N, level ** N):
            raise Exception("Incorret Coupling Matrix size")
    for d in drives:
        if np.shape(d)[0] != np.shape(d)[1]: raise Exception("Incorrect drive dimensions. Must be square.")
        if np.shape(d)[0] != level: raise Exception("All drives must the qudit size.")
    

    #PyTorch Parameter Optimization 
    if level >= 4 and CTLBool:
        if warmStartBool:
            try:
                R = torch.tensor(pd.read_csv(weightFName,header=None).to_numpy(), dtype=torch.double)
            except:
                R = torch.rand([M,len(drives)*N], dtype=torch.double) *2*np.pi
        else:
            R = torch.rand([M,8], dtype=torch.double) *2*np.pi
    else:
        if warmStartBool:
            try:
                R = torch.tensor(pd.read_csv(weightFName,header=None).to_numpy(), dtype=torch.double)
            except:
                R = torch.rand([M,len(drives)*N], dtype=torch.double) *2*np.pi
        else:
            R = torch.rand([M,len(drives)*N], dtype=torch.double) *2*np.pi
    R.requires_grad = True # set flag so we can backpropagate

    # Supports common optimizers 
    if optimizer == "SGD":
        optimizer = torch.optim.SGD([R], lr = 0.3, momentum=0.99, nesterov=True)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.03, factor=0.3, patience= 20)
    elif optimizer == "ADAM":
        optimizer = torch.optim.Adam([R], lr = 0.3)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.03, factor=0.3, patience= 20)
    elif optimizer == "CosineLR":
        optimizer = torch.optim.SGD([R], lr = 0.3, momentum=0.99, nesterov=True)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)

    # Generating time evolution and optimizing
    for n in range(0,N_iter):
        #Creating Drive Hamilontian
        U_Exp = 1
        for i in range(0,N):
            U_Exp = tensor(kron(U_Exp,id),dtype=dt)#initializing unitary
        
        if mlbool: #minimize leakage 
                #Initializing higher energy state occupancy values
                l = level - 1
                qttArr = stateProj(3,l)
                qttStates = vecSpaceGen(qttArr)
                qttOccVals = []
                
                #Qubit States
                qbArr = stateProj(1,l) + stateProj(0,l)
                qbStates = vecSpaceGen(qbArr)

                #Initial occupancy of higher energy state
                qttOccup = 0
                for state in qbStates: 
                    qttOccup += spaceOcc(U_Exp,state,qttStates)
                qttOccVals.append(qttOccup/len(qbStates))

        if level >= 4 and CTLBool: #Cross talk leakage modeling. Takes different refernce frame, so we run a different optimziation. 
            for m in range(M):
                pc = R[m]
                def CTL_H(t):
                    HD = torch.zeros((level,level),dtype=dt)
                    for mul,l in enumerate(range(1,level)):
                        if mul == 1:
                            HD[l,l] = anharmVal
                        else:
                            HD[l,l] = (mul*anharmVal)
                    H1 = HD.clone()
                    H2 = HD.clone()

                    if ContPulse == "True":
                        shape = tensor((np.sin(np.pi * t * M / tmin)) ** 2)
                        D1 = shape*(cp(t,pc[0],pc[4]) + cp(t,pc[1],pc[5],anharmVal) + cp(t,pc[2],pc[6],stag) + cp(t,pc[3],pc[7],stag + anharmVal))
                        D2 = shape*(cp(t,pc[1],pc[5]) + cp(t,pc[3],pc[7],anharmVal) + cp(t,pc[0],pc[4],-1*stag) + cp(t,pc[2],pc[6],-1*stag + anharmVal))
                    else:
                        D1 = cp(t,pc[0],pc[4]) + cp(t,pc[1],pc[5],anharmVal) + cp(t,pc[2],pc[6],stag) + cp(t,pc[3],pc[7],stag + anharmVal)
                        D2 = cp(t,pc[1],pc[5]) + cp(t,pc[3],pc[7],anharmVal) + cp(t,pc[0],pc[4],-1*stag) + cp(t,pc[2],pc[6],-1*stag + anharmVal)

                    for i in range(len(HD)-1):
                        H1[i,i+1] = D1
                        H1[i+1,i] = D1.conj()

                        H2[i,i+1] = D2
                        H2[i+1,i] = D2.conj()
                    if ContHBool: H = torch.tensor(H0(t)) + torch.kron(H1,torch.tensor(id)) + torch.kron(torch.tensor(id),H2)
                    else: H = H0 + torch.kron(H1,torch.tensor(id)) + torch.kron(torch.tensor(id),H2)

                    return H

                if ode == "RK2": U_Exp = normU(RK2(m/M*tmin,(m+1)/M*tmin,U_Exp,h,dUdt,CTL_H))
                elif ode == "SRK2": U_Exp = SRK2(m/M*tmin,(m+1)/M*tmin,U_Exp,h,CTL_H)
                else: raise Exception("Incorrect Cross Talk Modeling Type. Either Second Order Runge-Kutta, Störmer-Verlet, or symplectic Runge-Kutta.")

                if mlbool: #Calculating higher energy state occupancy
                    qttOccup = 0
                    for state in qbStates: 
                        qttOccup += spaceOcc(U_Exp,state,qttStates)
                    qttOccVals.append(qttOccup/len(qbStates))
                        
            #Here we need to define a two drives for each qudit. These two drives will be X and Y variants 
            # that include the cross-talk, leakage, drives, and internal inteference. 
        else: #Non cross-talk leakage modeling 
            # Here we will build our Hamiltonian into three parts. 
            # 1. Create a product of pulses to optimize on. These may be continuous or square 
            # 2. Create our coupling Hamiltonian. This may be time independent or dependent 
            # 3. Add Cross-Talk pulses if necessary
            for m in range(0,M):#1. Product of pulses
                timeDepen = False #time dependent bool
                pulse_coef = R[m]
                if ContBool: #Continuous (sin^2(x)) pulses
                    timeDepen = True
                    def H1(t):
                        HD = 0
                        for i,d in enumerate(drives):
                            HD = HD + sum_pauli(pulse_coef[i*N:(i+1)*N],d,t)
                            if lbool and i >= level -2:
                                    H1 = H1 + sum_pauli(pulse_coef[i*N:(i+1)*N],quditDrives[(i % len(quditDrives))],t)
                        if lbool: HD = HD + sum_pauli(tensor([1]*N),anharm,t)
                        return HD
                else: #Square Pulses
                    H1 = 0
                    for i,d in enumerate(drives):
                        H1 = H1 + sum_pauli(pulse_coef[i*N:(i+1)*N],d)
                        if lbool and i >= level -2:
                            H1 = H1 + sum_pauli(pulse_coef[i*N:(i+1)*N],quditDrives[(i % len(quditDrives))])
                    if lbool: 
                        #H1 = H1 + sum_pauli(tensor([1]*N),anharm) # different way to account for leakage 
                        H1 = H1 + tensor(kron(id,anharm) + kron(anharm,id))
                
                #Cross Talk
                if ctBool:
                    def H1ct(t):
                        if ContHBool: HDct = torch.zeros((len(H0(1)),len(H0(1))))
                        else: HDct = torch.zeros((len(H0),len(H0)))
                        for i,d in enumerate(drives):
                            HDct = HDct + sum_pauli(flip(pulse_coef[i*N:(i+1)*N],dims=[0]),d,t,stag)  #Cross talk drives with time dependent phases and fliped coefficients. 
                            if lbool:  
                                HDct = HDct + sum_pauli(flip(pulse_coef[i*N:(i+1)*N],dims=[0]),quditDrives[i],t,stag+anharmVal)  #remove anharmonicty due to interaction picture
                        HDct = HDct + HDct.conj().T # Hermitian Conjugate
                        return HDct
                    if timeDepen: 
                        H1 = lambda t: H1(t) + H1ct(t)
                    else: 
                        timeDepen = True
                        H1Cont = lambda t: H1 + torch.tensor(H1ct(t)) 
                if timeDepen or ContHBool:
                    # Adding static Hamiltonian
                    if ContHBool and timeDepen: 
                        if ctBool: H = lambda t: torch.tensor(H0(t)) + H1Cont(t)
                        else: H = lambda t: torch.tensor(H0(t)) + H1(t)
                    elif not ContHBool and timeDepen : H = lambda t: H0 + H1(t) #Need to fix this
                    elif ContHBool and not timeDepen: H = lambda t: torch.tensor(H0(t)) + H1

                    # Time dependent evolution
                    if ode == "RK2": U_Exp = normU(RK2(m/M*tmin,(m+1)/M*tmin,U_Exp,h,dUdt,H))
                    elif ode == "SRK2": U_Exp = SRK2(m/M*tmin,(m+1)/M*tmin,U_Exp,h,H)

                else:
                    #Time independent Evolution 
                    H = H0 + H1
                    U_Exp = matmul(matrix_exp(-1j*(H)*tmin/M),U_Exp)

                if mlbool: #Calculating higher energy state occupancy
                    qttOccup = 0
                    for state in qbStates: 
                        qttOccup += spaceOcc(U_Exp,state,qttStates)
                    qttOccVals.append(qttOccup/len(qbStates))

        #Fidelity calulcation given by Nielsen Paper
        fidelity = 0
        # d = 2**N

        SU = genTwoQuditBasis(dspaceLen,level,dt)

        # for U in SU:
        #     eps_U = matmul(matmul(U_Exp,U),(U_Exp.conj().T))
        #     target_U = matmul(matmul(input_gate,(U.conj().T)),(input_gate.conj().T))
        #     tr = trace(matmul(target_U,eps_U))
        #     fidelity = fidelity + tr 
        # fidelity = abs(fidelity + d*d)/(d*d*(d+1))    

        fid = 0
        for U in SU:
            eps_U = torch.matmul(torch.matmul(U_Exp,U),(U_Exp.conj().T))
            target_U = torch.matmul(torch.matmul(input_gate,(U.conj().T)),(input_gate.conj().T))
            tr = torch.trace(torch.matmul(target_U,eps_U))
            fid = fid + tr 
        d2 = dspaceLen**N
        fidelity = abs((fid +d2**2)/((d2**2) * (d2 + 1)))
        infidelity = 1 - fidelity
        infidelity_list[n] = infidelity.detach()
        weight_list.append(R.detach().numpy())

        if mlbool:
            #minimization type: let's start with average let alpha = 0.2
            mean = 0
            for qt in qttOccVals:
                mean += qt
            mean /= len(qttOccVals)
            max = torchMax(qttOccVals)
            cost = infidelity + alpha*(mean + max)
            cost.backward(retain_graph=True)
        else:
            infidelity.backward(retain_graph=True)

        #Printing statement
        #if (n+1)%1==0: print('Fidelity ', str(n+1), ' out of ', str(N_iter), 'complete. Avg Fidelity: ', str(1-infidelity.item()))

        #optimizer 
        optimizer.step()
        scheduler.step(infidelity)
        optimizer.zero_grad()

    #If warm start, then save the last iteration of weights.
    maxFid = 1 - infidelity_list.min().item()
    if warmStartBool and not wsmBoolF: 
        return [maxFid,R.detach().numpy()] #returning latest weigthts to put into next iteration 

    return [maxFid, weight_list[np.argmin(infidelity_list)]] #returning maximum fidelity and corresponding weights
    


