import numpy as np
import torch
from itertools import permutations
from itertools import product
import os
import random

def Qutrit_ML(J,B,M,input_gate,t,N_iter,pulse_file):
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Mon Aug 16 4:33 2021

    @author: Bora & Alex


    DESC: This function is essentially the qubit optimization done by Alex & I in Joel's experiment. But now we have drives from the |0> -> |1> and |1> -> |2>. 
    We still are generating 2-qubit gates, but the inclusion of qutrit drives minimizes the generation time. See PRA paper promising initial results.  

    UPDATE: Fixed Coupling and retesting removal of two state couping
    """
    #imports

    #Experimental Parameters
    g = 1
    maxFreq = 5*g


    #Paul Matrices in Qubtrit Space
    sxq = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) 
    syq = np.array([[0,-1j, 0],[1j,0, 0], [0, 0, 1]]) 
    sxxq = np.array([[1,0,0],[0,0,1],[0,1,0]]) 
    syyq = np.array([[1,0,0],[0,0,-1j],[0,1j,0]]) 

    #Pauli Matricies in Qubit Subspace
    sx = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) 
    sy = np.array([[0,-1j, 0],[1j,0, 0], [0, 0, 0]]) 
    sz = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) 
    id = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 0]]) 

    
    #Function definitions 

    def zero_mat(N):#Generates matrix of zeros
        zero_gate = np.array([[0,0,0],[0,0,0],[0,0,0]])
        init = zero_gate
        if N < 2:
            return 1
        for i in range(0,N - 1):
            zero_gate = torch.tensor(np.kron(zero_gate,init))
        return zero_gate
    
    def sum_pauli(coef, gate):#Sums Pauli gates with coefficients 
        N = len(coef)
        total_pauli = zero_mat(N)
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
    N = len(B)
    torch.manual_seed(random.randint(0,1000))
    dt = torch.cdouble # datatype and precision
    infidelity_list=torch.zeros([N_iter,1])

    #J coefficients gathering, essentially flattens the array
    J_coef = []
    for i in range(0,len(J) - 1):
        for j in range(0,len(J) - i - 1):
            J_coef.append(J[i,j].item())


    #H0 Generation
    H0 = torch.tensor(zero_mat(N),dtype=torch.cdouble)
    eigen_energies = [0, 5.440, 10.681, 4.994, 10.433, 15.666, 9.832, 15.270, 20.506] # experimentally given by Ray's Group
    for i,e in enumerate(eigen_energies):
        H0[i,i] = float(e)

    #Coupling terms in Hamiltonian (g1(|01><10| + h.c.) + g2(|12><21| + h.c.)
    g1 = g #What are the values fo g1 are g2? 
    g2 = 0
    zero_state = np.array([1,0,0])
    one_state = np.array([0,1,0])
    two_state = np.array([0,0,1])
    one_transition = np.kron(np.outer(zero_state,one_state),np.outer(one_state,zero_state))
    one_transition = one_transition + one_transition.conj().T
    two_transition = np.kron(np.outer(one_state,two_state),np.outer(two_state,one_state))
    two_transition = two_transition + two_transition.conj().T
    H0 = H0 + g1*one_transition + g2*two_transition

    #Unitary group generation
    SU = []
    pauli_int = [1,2,3,4]
    perms = list(product(pauli_int,repeat=N))#all permutations of paulis
    #Paul Matrices only in the qubit space 
    for p in perms:#mapping integers to pauli 
        unitary = 1
        for pauli in p:
            if pauli == 1:
                unitary = torch.tensor(np.kron(unitary,sx),dtype=torch.cdouble)
            elif pauli == 2:
                unitary = torch.tensor(np.kron(unitary,sy),dtype=torch.cdouble)
            elif pauli == 3:
                unitary = torch.tensor(np.kron(unitary,sz),dtype=torch.cdouble)
            elif pauli == 4:
                unitary = torch.tensor(np.kron(unitary,id),dtype=torch.cdouble)
        SU.append(unitary)

    #These are the coefficients we are optimizing
    R = torch.rand([M,4*N], dtype=torch.double) *2*np.pi # Random initialization (between 0 and 2pi)
    R.requires_grad = True # set flag so we can backpropagate

    #Optimizer settings(can be changed & opttimized)
    lr=0.3#learning rate

    opt = 'SGD'  # Choose optimizer - ADAM, SGD (typical). ADAMW, ADAMax, Adadelta,  
                        # Adagrad, Rprop, RMSprop, ASGD, also valid options.     
    sched = 'Plateau'  # Choose learning rate scheduler - Plateau, Exponential (typical), Step
    
    if opt=='ADAM': optimizer = torch.optim.Adam([R], lr = lr, weight_decay=1e-6)
    elif opt=='ADAMW': optimizer = torch.optim.AdamW([R], lr = lr, weight_decay=0.01)
    elif opt=='ADAMax': optimizer = torch.optim.Adamax([R], lr = lr, weight_decay=0.01)
    elif opt=='RMSprop': optimizer = torch.optim.RMSprop([R], lr = lr, momentum=0.2)
    elif opt=='Rprop': optimizer = torch.optim.Rprop([R], lr = lr)
    elif opt=='Adadelta': optimizer = torch.optim.Adadelta([R], lr = lr) 
    elif opt=='Adagrad': optimizer = torch.optim.Adagrad([R], lr = lr)
    elif opt=='SGD': optimizer = torch.optim.SGD([R], lr = lr, momentum=0.99, nesterov=True)
    elif opt=='ASGD': optimizer = torch.optim.ASGD([R], lr = lr)
    else: optimizer=None; opt='None'
        
    if sched=='Step': scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=N_iter/10, gamma=0.9)
    elif sched=='Exponential': scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    elif sched=='Plateau': scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.03, factor=0.3 , patience= 20 ); loss_in=True; 
    else: scheduler=None; sched='None'

    prev_infidelity = -1
    change_count = 0
    for n in range(0,N_iter):
        #Creating Drive Hamilontian
        U_Exp = 1
        for i in range(0,N):
            U_Exp = torch.tensor(np.kron(U_Exp,id),dtype=dt)#initializing unitary
        for m in range(0,M):#Product of pulses
            pulse_coef = R[m]
            H1 = sum_pauli(pulse_coef[:N],sxq) + sum_pauli(pulse_coef[N:2*N],syq) + sum_pauli(pulse_coef[2*N:3*N],sxxq) + sum_pauli(pulse_coef[3*N:4*N],syyq) 
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
        #if (n+1)%50==0: 
        #    print('Itertation ', str(n+1), ' out of ', str(N_iter), 'complete. Avg Infidelity: ', str(infidelity.item()))

        #optimizer 
        if optimizer is not None and scheduler is None:  # Update R
            optimizer.step()
            optimizer.zero_grad()
        elif optimizer is not None and scheduler is not None:
            optimizer.step()
            if loss_in: 
                scheduler.step(infidelity)
            else: 
                scheduler.step()
            optimizer.zero_grad()
        else:
            R.data.sub_(lr*R.grad.data) # using data avoids overwriting tensor object
            R.grad.data.zero_()           # and it's respective grad info
        
        #Stopping Condition for a lack of change  
        curr_infidelity = infidelity_list[n]
        if prev_infidelity == curr_infidelity and n != 0:
            change_count += 1
        else:
            change_count = 0
        if change_count == 100:
           return infidelity_list.min().item()
        prev_infidelity = curr_infidelity
        if 1 - infidelity_list[n] >= 99.99: #Stopping condition for high fidelity iterations
            return infidelity_list.min().item()

    tmin = np.pi/4

    pulse_file_time = pulse_file + "_time" + str(t/tmin)+"_"
    np.savetxt(os.path.join(os.getcwd(),"Pulse_Sequences/"+pulse_file+"/" +pulse_file_time+".csv"),R.detach().numpy(),delimiter=",") 
    return infidelity_list.min().item()