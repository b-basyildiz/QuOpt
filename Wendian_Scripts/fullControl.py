import sys
import numpy as np 
from numpy import array,kron,pi,sqrt
import os
from Full_ML import fidelity_ml
from itertools import permutations
from random import randint

#Input from Control Manger
quditType = str(sys.argv[1])
gateType = str(sys.argv[2])
couplingType = str(sys.argv[3])
segmentCount = int(sys.argv[4])
drivesType = str(sys.argv[5]) #all, qutrits, y
g = float(sys.argv[6])#coupling strength
rsCount = int(sys.argv[7])#random seed count
tNum = int(sys.argv[8])#number of points
iterationCount = int(sys.argv[9])#number of iterations
maxDriveStrength = int(sys.argv[10])
maxTime = float(sys.argv[11]) #maximum time (T/Tmin)
t = float(sys.argv[12])/tNum # Input is [0,..,number of points]

#THINGS TO CHANGE WHEN TESTING: random seed count -> 50, iterations -> 5,000
#rs_count = 50
#iterationCount = 1
print_statements = False
Diagonal = False
#mainDir = "ML_Output"
mainDir = "Data"


#Static Parameters
Fidelities = []
Times = []
drives = []
tgate = None
H0 = None
tmin = None
if quditType == "Qubit": level = 2
elif quditType == "Qutrit": level = 3
else: raise Exception("Incorrect qudit. Either qubit, qutrit, or quartit ")

#File creation
fname = quditType + "_" + gateType + "_" + couplingType + "_M" + str(segmentCount)+ "_" + drivesType + "_g" + str(g) + "_maxT" + str(maxTime)
if maxDriveStrength != -1: fname = fname + "_maxD" + str(maxDriveStrength)

#Gate Workflow
if gateType == "CNOT":
    if level == 2: tgate = array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    else: tgate = array([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]])
elif gateType == "iSWAP":
    if level == 2: tgate = array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]])
    else: tgate = array([[1,0,0,0,0,0,0,0,0],[0,0,0,1j,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,1j,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]])
elif gateType == "SWAP": 
    if level == 2: tgate = array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    else: tgate = array([[1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]])
elif gateType == "iTwoPhonon":
    tgate = array([[0,0,0,0,-1j,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[-1j,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]],dtype=complex)
else: raise Exception("Invalid Gate Type. Enter CNOT, iSWAP, or SWAP.")

#Speed limit Workflow
 #Needs to be change for different couplings, but do in the future
if gateType == "CNOT":
    tmin = pi/4
elif gateType == "iSWAP":
    tmin = pi/2
elif gateType == "SWAP":
    tmin = 3*pi/4
elif gateType == "iTwoPhonon":
    tmin = pi/2

#Coupling Workflow
if couplingType == "XX":
    if level == 2: 
        sx = array([[0, 1], [1, 0]])
        H0 = kron(sx,sx)
    elif level ==3: 
        sx = array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) 
        sxx = array([[0,0,0],[0,0,1],[0,1,0]])
        H0 = kron(sx,sx) + kron(sxx,sxx)
elif couplingType == "Ashhab":
    annhilate = array([[0,1,0],[0,0,np.sqrt(2)],[0,0,0]])
    create = annhilate.T
    H0 = kron(annhilate + create,annhilate + create)
elif couplingType == "AshhUnit":
    annhilate = array([[0,1,0],[0,0,1],[0,0,0]])
    create = annhilate.T
    H0 = kron(annhilate + create,annhilate + create)
# elif couplingType == "AshhabHopp":
#     annhilate = array([[0,1,0],[0,0,np.sqrt(2)],[0,0,0]])
#     create = annhilate.T
#     H0 = kron(annhilate,create) + kron(create,annhilate)
# elif couplingType == "AshhabLabFrame":
#     #Couplings Terms
#     annhilate = array([[0,1,0],[0,0,np.sqrt(2)],[0,0,0]])
#     create = annhilate.T
#     H0 = kron(annhilate + create,annhilate + create)
#     #Diagonal Entries 
#     diagEntries = [0, 5.440, 10.681, 4.994, 10.433, 15.666, 9.832, 15.270, 20.506]
#     for i,d in enumerate(diagEntries):
#         H0[i,i] = d
# elif couplingType == "CnotProtocol":
#     H = np.zeros([3 ** 2, 3 ** 2])
#     H[3,4] = 1
#     H[3,5] = 1
#     H0 = H + H.transpose()
#     tmin = np.pi/2
# elif couplingType == "iSwapProtocol":
#     H = np.zeros([3 ** 2, 3 ** 2])
#     H[1,3] = 1
#     H[2,3] = 1
#     H0 = H + H.transpose()
elif couplingType == "Analy":
    H0 = np.zeros([3 ** 2, 3 ** 2])
    H0[0,4] = 1
    H0[2,4] = sqrt(2)
    H0[6,4] = sqrt(2)
    H0[8,4] = 2
    H0 = H0 + H0.transpose()
# elif couplingType[:8] == "AnalyNeg":
#     vals = [1,-1] * 3
#     negList = list(permutations(vals,4))
#     perm = int(couplingType[8:])
#     negs = negList[perm]
#     H0 = np.zeros([3 ** 2, 3 ** 2])

#     H0[0,4] = negs[0]*1
#     H0[2,4] = negs[1]*sqrt(2)
#     H0[6,4] = negs[2]*sqrt(2)
#     H0[8,4] = negs[3]*2

#     H0 = H0 + H0.transpose()
#     couplingType = couplingType[:8]
# elif couplingType == "AllCouplings":
#     H0 = np.zeros([3 ** 2, 3 ** 2])
#     for i in range(9):
#         if i != 4: H0[i,4] = 1
#     H0 = H0 + H0.transpose()
# elif couplingType == "Diagonal":
#     H0 = np.zeros([3 ** 2, 3 ** 2])
#     Diagonal = True
# elif couplingType == "AllCouplingsDiag":
#     H0 = np.ones([level ** 2, level ** 2])
#     for i in range(level ** 2):
#         H0[i,i] = 0
else: raise Exception("Incorrect Coupling Type. (XX, Ashabb, AshhabOnes, AshhabHopp, AshhabbLabFrame, CnotProtocol, iSwapProtocol, AnalyticalSpeedUp, AllCouplings, Diagonal, AllCouplingsDiag)")
H0 = g*H0

#Drives workflow
if drivesType == "all":
    if level == 2: #Qubit Subspace
        sx = array([[0, 1], [1, 0]])
        sy = array([[0,-1j],[1j,0]])
        drives.append(sx)
        drives.append(sy)
    elif level == 3: #Qutrit Subspace
        sx = array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) 
        sy = array([[0,-1j, 0],[1j,0, 0], [0, 0, 0]]) 
        sxx = array([[0,0,0],[0,0,1],[0,1,0]]) 
        syy = array([[0,0,0],[0,0,-1j],[0,1j,0]]) 
        drives.append(sx)
        drives.append(sy)
        drives.append(sxx)
        drives.append(syy)
elif drivesType == "qtd":
    sxx = array([[0,0,0],[0,0,1],[0,1,0]]) 
    syy = array([[0,0,0],[0,0,-1j],[0,1j,0]]) 
    drives.append(sxx)
    drives.append(syy)
elif drivesType == "yd":
    syy = array([[0,0,0],[0,0,-1j],[0,1j,0]]) 
    drives.append(syy)
elif drivesType == "dipoleAll":
    sx = array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) 
    sy = array([[0,-1j, 0],[1j,0, 0], [0, 0, 0]]) 
    sxx = array([[0,0,1],[0,0,0],[1,0,0]]) 
    syy = array([[0,0,-1j],[0,0,0],[1j,0,0]]) 
    drives.append(sx)
    drives.append(sy)
    drives.append(sxx)
    drives.append(syy)
elif drivesType == "dipoleQtd":
    sxx = array([[0,0,1],[0,0,0],[1,0,0]]) 
    syy = array([[0,0,-1j],[0,0,0],[1j,0,0]]) 
    drives.append(sxx)
    drives.append(syy)
else: raise Exception("Incorrect amount of drives (all, qtd, yd, dipoleAll, dipoleqtd)")

#Directory Creation 
try: #All files are stored under their gateType
    os.makedirs(mainDir)
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
if Diagonal:
    try:
        deDir = os.path.join(mainDir, "DiagonalEntries")
        os.makedirs(deDir)
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
for s in seeds:
    if not Diagonal:[fidelity,W] = fidelity_ml(segmentCount,tgate,t*tmin*maxTime,iterationCount,s,H0,drives,maxDriveStrength) #3 segments, given time *tmin, 5000 iterations, s random seed
    else: [fidelity,W,dentries] = fidelity_ml(segmentCount,tgate,t*tmin*maxTime,iterationCount,s,H0,drives,maxDriveStrength)
    if print_statements:print("The fidelity for seed " + str(s) + " for time t=" + str(t) + " is: " + str(fidelity))
    if fidelity > max_fidelity:
        max_fidelity = fidelity
        fWname = fname + "_Wt" + str(round(t*maxTime,2)) + ".csv"
        fWname = os.path.join(gDir, fWname)
        np.savetxt(fWname,W,delimiter=",")
        if Diagonal: 
            fDEname = fname + "_DE" + str(round(t*maxTime,2)) + ".csv"
            fDEname = os.path.join(deDir, fDEname)
            np.savetxt(fDEname,dentries,delimiter=",")
out_arr = np.array([[max_fidelity,round(t*maxTime,2)]]) #File output 
#fname = os.path.join(mainDir, fname + ".csv")
fname = os.path.join(fDir, fname + ".csv")
if print_statements:print("The maximum fidelity for time t=" + str(round(t*maxTime,2)) + " is: " + str(max_fidelity),end="\n\n")
with open(fname, 'a') as file:
    np.savetxt(file,out_arr,delimiter=",")

