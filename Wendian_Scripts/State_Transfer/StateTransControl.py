import sys
import numpy as np 
import torch 
import os

#THINGS TO CHANGE WHEN TESTING: random seed count -> 50, iterations -> 5,000
rs_count = 50
iteration_count = 5000
mainDir = "StateTrans"

#Input from Control Manger
mlType = str(sys.argv[1])
couplingType = str(sys.argv[2])
t = float(sys.argv[3])/20 #20 is the number of points. Input is [0,..,20]

#Static Parameters
dt = torch.cdouble #data type 
Fidelities = []
Times = []
tmin = np.pi

#Picking Specific ML (condense all ML into a singular modal file later)
H0 = None
if mlType == "Qubit": 
    from QB_StateTrans import stateTransfer_ml
    H0 = torch.tensor(np.kron(np.array([[0,0],[0,1]]),np.array([[0,0],[0,1]])),dtype=dt)
    fname = mlType
elif mlType == "Qutrit": 
    from QT_StateTrans import stateTransfer_ml
    fname = couplingType + "_" + mlType
    if couplingType == "num_op":
        H0 = torch.tensor(np.kron(np.array([[0,0,0],[0,1,0],[0,0,2]]),np.array([[0,0,0],[0,1,0],[0,0,2]])),dtype=dt)
    elif couplingType == "ones":
        H0 = torch.tensor(np.kron(np.array([[0,0,0],[0,1,0],[0,0,1]]),np.array([[0,0,0],[0,1,0],[0,0,1]])),dtype=dt)
    elif couplingType == "normalized":
        H0 = torch.tensor(1/4 * np.kron(np.array([[0,0,0],[0,1,0],[0,0,2]]),np.array([[0,0,0],[0,1,0],[0,0,2]])),dtype=dt)
    else: raise Exception("Invalid Coupling Type! Valid types are \'num_op\','ones','normalized'")
else: raise Exception("Invalid Machine Learning Type! Valid types are 'Qubit' or 'Qutrit'")

#print(mlType + ", " + couplingType)
# print(H0)

#File Creation 
try:
    os.makedirs(mainDir)
except:
    pass 
try: #Making Gate Folder
    gDir = os.path.join(mainDir, couplingType)
    os.makedirs(gDir)
except:
    pass
try:
    wDir = os.path.join(gDir,mlType + "_Weights")
    os.makedirs(wDir)
except:
    pass


#Random Seed averagging 
max_fidelity = 0
seeds = np.random.randint(0,100,size=rs_count)
for s in seeds:
    [fidelity,W] = stateTransfer_ml(16,t*tmin,iteration_count,s,H0) #16 segments, given time *tmin, 5000 iterations, s random seed
    #print("The fidelity for seed " + str(s) + " for time t=" + str(t*tmin) + " is: " + str(fidelity))
    if fidelity > max_fidelity:
        max_fidelity = fidelity
        fWname = fname + "_Wt" + str(t) + ".csv"
        fWname = os.path.join(wDir, fWname)
        np.savetxt(fWname,W,delimiter=",")
out_arr = np.array([[max_fidelity,t]]) #File output 
fname = os.path.join(gDir, fname + ".csv")
#print("The maximum fidelity for time t=" + str(t) + " is: " + str(max_fidelity),end="\n\n")
with open(fname, 'a') as file:
    np.savetxt(file,out_arr,delimiter=",")

