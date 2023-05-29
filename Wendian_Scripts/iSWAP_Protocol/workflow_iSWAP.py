import sys
import numpy as np 
import torch 
import os

#THINGS TO CHANGE WHEN TESTING: random seed count -> 50, iterations -> 5,000
rs_count = 50
iteration_count = 5000
print_statements = False
mainDir = "iSWAP_Protocol"


#Input from Control Manger
#subspace = str(sys.argv[1]) #all, qutrits, y_qutrit
level = int(sys.argv[1])
t = float(sys.argv[2])/20 #20 is the number of points. Input is [0,..,20]

#Static Parameters
dt = torch.cdouble #data type 
tgate = None
g = None
Fidelities = []
Times = []
if level == 2: fname = "Qubit"
elif level == 3: fname = "Qutrit"
else: raise Exception("Incorrect qudit. Either qubit or qutrit ")

#Picking Specific ML (condense all ML into a singular modal file later)
from Full_ML import fidelity_ml
#iSWAP = torch.tensor([[1,0,0,0,0,0,0,0,0],[0,0,0,1j,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,1j,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]],dtype=dt)
CNOT = torch.tensor([[1,0,0,0,0,0,0,0,0],[0,0,0,1j,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,1j,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]],dtype=dt)
tmin = np.pi/4 #np.pi/2 for iSWAP

#print(mlType + ", " + gateType +"\n" + str(tgate.detach().numpy()),end=": \n\n")

#File Creation 
try:
    os.makedirs(mainDir)
except:
    pass 
try:
    wDir = os.path.join(mainDir,fname + "_Weights")
    os.makedirs(wDir)
except:
    pass
#print(fname)

#Coupling Hamiltonain
H = np.zeros([3 ** 2, 3 ** 2])
H[1,3] = 1
H[2,3] = 1
H0 = torch.tensor(H + H.transpose())

#Subspace sorting
# sub_num = -1
# if subspace == "all": sub_num = 0
# elif subspace == "qutrits": sub_num = 1
# elif subspace == "y_qutrit": sub_num = 2
# else: raise Exception("Incorret Subspace Input (all,qutrits,y_qutrit)")
#Random Seed averaging 
max_fidelity = 0
seeds = np.random.randint(0,100,size=rs_count)
for s in seeds:
    [fidelity,W] = fidelity_ml(3,iSWAP,t*tmin,iteration_count,s,H0,3,0) #3 segments, given time *tmin, 5000 iterations, s random seed
    if print_statements:print("The fidelity for seed " + str(s) + " for time t=" + str(t) + " is: " + str(fidelity))
    if fidelity > max_fidelity:
        max_fidelity = fidelity
        fWname = fname + "_Wt" + str(t) + ".csv"
        fWname = os.path.join(wDir, fWname)
        np.savetxt(fWname,W,delimiter=",")
out_arr = np.array([[max_fidelity,t]]) #File output 
fname = os.path.join(mainDir, fname + ".csv")
if print_statements:print("The maximum fidelity for time t=" + str(t) + " is: " + str(max_fidelity),end="\n\n")
with open(fname, 'a') as file:
    np.savetxt(file,out_arr,delimiter=",")

