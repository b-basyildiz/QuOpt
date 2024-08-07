import sys
import numpy as np 
from numpy import zeros
import os
from ML import fidelity_ml
from helperFuncs import *
import pandas as pd
from filelock import FileLock
import time

#Input from Control Manger
quditType = str(sys.argv[1])
gateType = str(sys.argv[2])

couplingType = str(sys.argv[3])
segmentCount = int(sys.argv[4])
g = float(sys.argv[5])#coupling strength
anharmonicity = float(sys.argv[6])

crossTalk = str(sys.argv[7])
staggering = float(sys.argv[8])#staggering of two qubits

ode = str(sys.argv[9])
h = float(sys.argv[10])
alpha = float(sys.argv[11])

ContPulse = str(sys.argv[12])
leakage = str(sys.argv[13]) #all, qutrits, y
minLeak = str(sys.argv[14])
maxDriveStrength = int(sys.argv[15])

minTime = float(sys.argv[16])
maxTime = float(sys.argv[17]) #maximum time (T/Tmin)
points = int(sys.argv[18])#number of points

#randomSeedCount = int(sys.argv[17])#random seed count
iterationCount = int(sys.argv[19])#number of iterations

#Optimzer type
optimizer=str(sys.argv[20])

#t = float(sys.argv[19])/points # Input is [1,..,number of points]
index = int(sys.argv[21])

#THINGS TO CHANGE WHEN TESTING: random seed count -> 50, iterations -> 5,000
print_statements = False
#mainDir = "ML_Output"
mainDir = "Data"

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
elif tempQuditType == "quatrit": level = 4
else: raise Exception("Incorrect qudit. Either qubit, qutrit, qubit(leakage), or qutrit(leakage)")

if leakage == "True": #leakge models the system with an additional energy level. 
    level += 1

#Gate Workflow
tgate = gateGen(gateType,level)

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
if couplingType != "ContH":
    H0 = g*genCouplMat(couplingType,level)
else: 
    Evec = genEvec(anharmonicity,anharmonicity,staggering,g)/g # we divide by the couplling strength as everything we do is in units of g
    H0 = lambda t: HC(t,Evec)

#Drives for single phonon transitions 
if leakage != "True":
    for l in range(1,level):
        drives.append(genDrive(level,l,"x"))
        drives.append(genDrive(level,l,"y"))
else:
    for l in range(1,level-1):
        drives.append(genDrive(level,l,"x"))
        drives.append(genDrive(level,l,"y"))

if leakage == "True":
    ldrives = []
    for l in range(1,level): 
        if l == level - 1:
            ldrives.append(genDrive(level,l,"x"))
            ldrives.append(genDrive(level,l,"y"))

    #anharm = zeros((level,level))
    #anharm[-1,-1] = anharmonicity
    anharmVals = array([anharmonicity*(i+1) for i in range(level-2)])
    anharm = zeros((level,level))
    for l in range(2,level):
         anharm[-1*(l-1),-1*(l-1)] = anharmVals[::-1][l-2]

    drives = [drives,ldrives,anharm]


#File creation
fname = quditType + "_" + gateType + "_" + couplingType + "_M" + str(segmentCount) + "_" + optimizer + "_" + ode
if leakage == "True": fname = fname + "_leakage"+ str(anharmonicity)
fname = fname + "_g" + str(g) + "_maxT" + str(maxTime)
#if anharmonicityType > 1: fname = fname + "_anharmType" + str(anharmonicityType)
if maxDriveStrength != -1: fname = fname + "_maxD" + str(maxDriveStrength)
if crossTalk != "False": fname = fname + "_CTh" + str(h) + "_stag" + str(staggering)
if ContPulse != "False": fname = fname + "_Cont" + str(ode) 
fname = fname + "_it" + str(iterationCount)
if minLeak == "True": fname = fname + "_leakMin" + str(alpha)

#Directory Creation 
try: #All files are stored under their gateType
    os.makedirs(mainDir)
except:
    pass 
if leakage == "True" and crossTalk == "True": #CTL Model
    try:
        mainDir = os.path.join(mainDir,"CTL")
    except:
        pass
elif leakage == "True": # Leakage Model
    try:
        mainDir = os.path.join(mainDir,"Leakage")
    except:
        pass
elif crossTalk == "True": # Cross-Talk Model
        try: 
            mainDir = os.path.join(mainDir,"Cross_Talk")
        except:
            pass
else: #Traditional Model
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
#max_fidelity = 0
#seeds = np.random.randint(0,100,size=randomSeedCount)
if points != -1:
    seed = np.random.randint(0,100)
    times = np.linspace(minTime,maxTime,points)
    t = times[index]
else:
    seed = 1
    t = maxTime

lbool = False
if leakage == "True":
    lbool = True
[fidelity,W] = fidelity_ml(segmentCount,tgate,t*tmin,iterationCount,seed,H0,drives,maxDriveStrength,lbool,minLeak,crossTalk,h,alpha,anharmonicity,staggering,ode,ContPulse,optimizer)

fname = os.path.join(fDir, fname + ".csv")
fWname = "Weights_t" + str(round(t,4)) + ".csv"
fWname = os.path.join(gDir, fWname)

flock = FileLock(fname + ".lock")
wlock = FileLock(fWname + ".lock")

def write():
    #Writing the fidelity
    out_arr = np.array([[fidelity,round(t,4)]]) #File output 
    with open(fname, 'a') as file:
        np.savetxt(file,out_arr,delimiter=",") #Fidelty writing
    np.savetxt(fWname,W,delimiter=",") #Weights writing
    wlock.release()
    flock.release()
    try:
        os.remove(fname + ".lock")
    except:
        pass
    try:
        os.remove(fWname + ".lock")
    except:
        pass
    exit()
#Writing to csv
flock.acquire()
wlock.acquire()
try: #if the file has been made
    fidels = pd.read_csv(fname,names=["fidelity","time"])
except: #if the fidelity file has not been made
    write()
if fidels["time"].isin([t]).any(): #fidelity for time has been previous caluclated <- look into this, we need to compare times not if they exist
    tempFid = float(fidels[fidels["time"] == t]["fidelity"])
    if fidelity > tempFid: #if our fidelity is greater than the previous fidelity
        fIndex = fidels[fidels["time"] == t].index.to_numpy()[0] #What row our fidelity is at in the file
        fidels.iloc[fIndex,0] = fidelity
        fidels.to_csv(fname,index=False,header=False) #overwritting the previous file
        np.savetxt(fWname,W,delimiter=",") #Weights writing
    wlock.release()
    flock.release()
    try:
        os.remove(fname + ".lock")
    except:
        pass
    try:
        os.remove(fWname + ".lock")
    except:
        pass
    exit()
else: #fidelity for time has not been written to
    write()

# np.savetxt(fWname,W,delimiter=",")