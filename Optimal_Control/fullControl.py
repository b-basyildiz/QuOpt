import sys
import numpy as np 
from numpy import array,kron,zeros
import os
from Full_ML import fidelity_ml
from itertools import permutations
from random import randint
from helperFuncs import *

#Input from Control Manger
quditType = str(sys.argv[1])
gateType = str(sys.argv[2])
couplingType = str(sys.argv[3])
segmentCount = int(sys.argv[4])
drivesType = str(sys.argv[5]) #all, qutrits, y
anharmonicity = float(sys.argv[6])
crossTalk = str(sys.argv[7])
g = float(sys.argv[8])#coupling strength
stag = float(sys.argv[9])#staggering of two qubits
rsCount = int(sys.argv[10])#random seed count
tNum = int(sys.argv[11])#number of points
iterationCount = int(sys.argv[12])#number of iterations
maxDriveStrength = int(sys.argv[13])
maxTime = float(sys.argv[14]) #maximum time (T/Tmin)
t = float(sys.argv[15])/tNum # Input is [0,..,number of points]

#THINGS TO CHANGE WHEN TESTING: random seed count -> 50, iterations -> 5,000
print_statements = False
leakage = False
#mainDir = "ML_Output"
mainDir = "Data"
ctVal = 0


#Static Parameters
Fidelities = []
Times = []
drives = []
tgate = None
H0 = None
tmin = None

# Energy level checking
tempQuditType= quditType.lower()
if tempQuditType == "qubit": level = 2
elif tempQuditType == "qutrit": level = 3
else: raise Exception("Incorrect qudit. Either qubit, qutrit, qubit(leakage), or qutrit(leakage)")

if drivesType == "leakage": #leakge models the system with an additional energy level. 
    level += 1

#Gate Workflow
tgate = gateGen(gateType,level)
# if gateType == "CNOT":
#     tgate = gateGen(gateType,level)
# elif gateType == "iSWAP":
#     tgate = gateGen(gateType,level)
# elif gateType == "SWAP": 
#     if level == 2: tgate = array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
#     else: tgate = array([[1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]])
# elif gateType == "iTwoPhonon":
#     tgate = array([[0,0,0,0,-1j,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[-1j,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]],dtype=complex)
# else: raise Exception("Invalid Gate Type. Enter CNOT, iSWAP, or SWAP.")

#Speed limit Workflow
 #Needs to be change for different couplings, but do in the future
if gateType == "CNOT":
    tmin = np.pi/4
elif gateType == "iSWAP":
    tmin = np.pi/2
elif gateType == "SWAP":
    tmin = 3*np.pi/4
elif gateType == "iTwoPhonon":
    tmin = np.pi/2

#Coupling Workflow
H0 = g*genCouplMat(couplingType,level)

#Drives workflow
if drivesType == "all":
    for l in range(1,level):
        drives.append(genDrive(level,l,"x"))
        drives.append(genDrive(level,l,"y"))
elif drivesType == "qtd":
    drives.append(genDrive(level,level-1,"x"))
    drives.append(genDrive(level,level-1,"y"))
elif drivesType == "yd":
    drives.append(genDrive(level,level-1,"y"))
elif drivesType == "twoPhonAll":
    for l in range(1,level):
        if l == 1:
            drives.append(genDrive(level,l,"x"))
            drives.append(genDrive(level,l,"y"))
        else:
            drives.append(genDrive(level,l,"tpx"))
            drives.append(genDrive(level,l,"tpy"))
elif drivesType == "twoPhonQtd":
    drives.append(genDrive(level,level-1,"tpx"))
    drives.append(genDrive(level,level-1,"tpy"))
elif drivesType == "leakage":
    leakage = True
    tempDrives = []
    tempLDrives = []

    for l in range(1,level): #something is wrong here
        if l == level - 1:
            tempLDrives.append(genDrive(level,l,"x"))
            tempLDrives.append(genDrive(level,l,"y"))
        else:
            tempDrives.append(genDrive(level,l,"x"))
            tempDrives.append(genDrive(level,l,"y"))

    #anharm = anharmVal*array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]) 
    # anharmVals = array([anharmonicity*(i+1) for i in range(level-2)])
    # anharm = zeros((level,level))
    # for l in range(2,level):
    #     anharm[-1*(l-1),-1*(l-1)] = anharmVals[::-1][l-2]

    anharm = zeros((level,level))
    anharm[-1,-1] = anharmonicity

    drives = [tempDrives,tempLDrives,anharm]
else: raise Exception("Incorrect amount of drives (all, qtd, yd, twoPhonAll, twoPhonQtd, leakage)")

#File creation
fname = quditType + "_" + gateType + "_" + couplingType + "_M" + str(segmentCount)+ "_" + drivesType 
if drivesType == "leakage": fname = fname + str(anharmonicity)
fname = fname + "_g" + str(g) + "_maxT" + str(maxTime)
if maxDriveStrength != -1: fname = fname + "_maxD" + str(maxDriveStrength)
if crossTalk != "False": fname = fname + "_CT" + str(crossTalk) + "_stag" + str(stag)

#Cross Talk Parsing 
if crossTalk != "False":
    if crossTalk[:3] == "ode":
        ctVal = float(crossTalk[3:])
        crossTalk = "ode"
    elif crossTalk[:4] == "disc":
        ctVal = int(crossTalk[4:])
        crossTalk = "disc"
    else: raise Exception("Incorrect way to model Cross Talk. Either ode or disc(rete).")

#Directory Creation 
try: #All files are stored under their gateType
    os.makedirs(mainDir)
except:
    pass 
if leakage:
    try:
        mainDir = os.path.join(mainDir,"Leakage")
    except:
        pass
else:
    try:
        mainDir = os.path.join(mainDir,"Closed_System")
    except:
        pass
try: #Each gate type has weights for a given qudit and coupling
    mainDir = os.path.join(mainDir, couplingType)
    os.makedirs(mainDir)
except:
    pass
try: #Each gate type has weights for a given qudit and coupling
    fDir = os.path.join(mainDir, "Fidelities")
    os.makedirs(fDir)
except:
    pass
try: #Each gate type has weights for a given qudit and coupling
    wDir = os.path.join(mainDir, "Weights")
    os.makedirs(fDir)
except:
    pass
try: #Each gate type has weights for a given qudit and coupling
    gDir = os.path.join(wDir, fname + "_Weights")
    os.makedirs(gDir)
except:
    pass


#Testing statements
# print(quditType)
# print("Gatetype: " + gateType)
# print(tgate)
# print("Tmin: " + str(tmin))
# print("CouplingType: " + couplingType)
# print(H0)
# print("SegmentNumber: " + str(segmentCount))
# print("Coupling strength: " + str(g))
# print("Drive Type: " + drives_type)
# for d in drives:
#     print(d)
# print("Time: " + str(t))

#Random Seed averaging 
max_fidelity = 0
seeds = np.random.randint(0,100,size=rsCount)
MCT = "False"
for s in seeds:
    #if not Diagonal:[fidelity,W] = fidelity_ml(segmentCount,tgate,t*tmin*maxTime,iterationCount,s,H0,drives,maxDriveStrength,leakage) #3 segments, given time *tmin, 5000 iterations, s random seed
    #else: [fidelity,W,dentries] = fidelity_ml(segmentCount,tgate,t*tmin*maxTime,iterationCount,s,H0,drives,maxDriveStrength,leakage)
    [fidelity,W] = fidelity_ml(segmentCount,tgate,t*tmin*maxTime,iterationCount,s,H0,drives,maxDriveStrength,leakage,crossTalk,ctVal,stag)
    if print_statements:print("The fidelity for seed " + str(s) + " for time t=" + str(t) + " is: " + str(fidelity))
    if fidelity > max_fidelity:
        max_fidelity = fidelity
        fWname = "Weights_t" + str(round(t*maxTime,2)) + ".csv"
        #fWname = fname + "_Wt" + str(round(t*maxTime,2)) + ".csv"
        fWname = os.path.join(gDir, fWname)
        np.savetxt(fWname,W,delimiter=",")
out_arr = np.array([[max_fidelity,round(t*maxTime,2)]]) #File output 
#fname = os.path.join(mainDir, fname + ".csv")
fname = os.path.join(fDir, fname + ".csv")
#fname = os.path.join(fDir, "Fidelities.csv")
if print_statements:print("The maximum fidelity for time t=" + str(round(t*maxTime,2)) + " is: " + str(max_fidelity),end="\n\n")
with open(fname, 'a') as file:
    np.savetxt(file,out_arr,delimiter=",")


