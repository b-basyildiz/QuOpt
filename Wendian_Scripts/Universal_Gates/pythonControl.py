import sys
import numpy as np 
import torch 
import os

#THINGS TO CHANGE WHEN TESTING: random seed count -> 50, iterations -> 5,000
rs_count = 50
iteration_count = 5000
#mainDir = "iSWAP_Protocol"

#Input from Control Manger
mlType = str(sys.argv[1])
gateType = str(sys.argv[2])
t = float(sys.argv[3])/20 #20 is the number of points. Input is [0,..,20]
mainDir = gateType + "_Protocol"

#Static Parameters
dt = torch.cdouble #data type 
tgate = None
g = None
Fidelities = []
Times = []
fname = gateType + "_" + mlType

#Picking Specific ML (condense all ML into a singular modal file later)
if mlType == "Qubit" or mlType == "Qubit_gRoot2": from randomSeed_ML import fidelity_ml
elif mlType == "Ashabb": from randomSeed_AML import fidelity_ml
elif mlType == "ZXG_Protocol_Sub": from ZXG_subProtocol_ML import fidelity_ml
elif mlType == "Qutrit": from ZXG_Protocol_ML import fidelity_ml
elif mlType == "Qubit_nop": from nop_QBML import fidelity_ml
elif mlType == "Qutrit_nop": from nop_QTML import fidelity_ml
elif mlType == "Qutrit_nop2": from nop_QTML2 import fidelity_ml
else: raise Exception("Invalid Machine Learning Type!")


if gateType == "CNOT" and mlType != "Qubit": tmin = np.pi # Note this is changed for the NOP protocol
elif gateType == "CNOT": tmin = np.pi/4
elif gateType == "iSWAP": tmin = np.pi/2
elif gateType == "SWAP": tmin = 3*np.pi/4
else: raise Exception("Invalid Gate Type!")

#Determing Qubit or Qutrit Gates
if mlType == "Qubit" or mlType == "Qubit_gRoot2" or mlType == "Qubit_nop":
    if gateType == "CNOT": tgate = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=dt)
    elif gateType == "iSWAP": tgate = torch.tensor([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]],dtype=dt)
    elif gateType == "SWAP": tgate =  torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=dt)
    else: raise Exception("Invalid ML and gate combination")

    if mlType == "Qubit" or mlType == "Qubit_nop":
        g = 1
    elif mlType == "Qubit_gRoot2":
        if gateType == "CNOT":
            g = 3
        else:
            g = np.sqrt(2)
    else: raise Exception("Invalid Qubit ML Type")
elif mlType == "Ashabb" or mlType == "ZXG_Protocol_Sub" or mlType == "Qutrit" or mlType == "Qutrit_nop" or mlType == "Qutrit_nop2":
    if gateType == "CNOT": tgate = torch.tensor([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]],dtype=dt)
    elif gateType == "iSWAP": tgate = torch.tensor([[1,0,0,0,0,0,0,0,0],[0,0,0,1j,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,1j,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]],dtype=dt)
    elif gateType == "SWAP": tgate = torch.tensor([[1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]],dtype=dt)
    else: raise Exception("Invalid ML and gate combination")
else:
    raise Exception("Invalid ML and gate combination")

#print(mlType + ", " + gateType +"\n" + str(tgate.detach().numpy()),end=": \n\n")

#File Creation 
try:
    os.makedirs(mainDir)
except:
    pass 
try: #Making Gate Folder
    gDir = os.path.join(mainDir, gateType)
    os.makedirs(gDir)
except:
    pass
try:
    wDir = os.path.join(gDir,mlType + "_Weights")
    os.makedirs(wDir)
except:
    pass
#print(fname)


#Random Seed averaging 
max_fidelity = 0
seeds = np.random.randint(0,100,size=rs_count)
for s in seeds:
    [fidelity,W] = fidelity_ml(16,tgate,t*tmin,iteration_count,g,s) #16 segments, given time *tmin, 5000 iterations, s random seed
    #print("The fidelity for seed " + str(s) + " for time t=" + str(t) + " is: " + str(fidelity))
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

