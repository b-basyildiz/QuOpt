import sys
import numpy as np 
from numpy import zeros
import os
from ML import fidelity_ml
from helperFuncs import *
import pandas as pd
from filelock import FileLock
import datetime
import hashlib

#Input from Control Manger
gateType = str(sys.argv[1])
level = int(sys.argv[2])

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
seed = int(sys.argv[22])

#Warm Start
warmStart=int(sys.argv[23])

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

# Qudit Type
quditType = ""
if level == 2: quditType = "qubit"
elif level == 3: quditType = "qutrit"
elif level == 4: quditType = "quatrit"
else: quditType = "qudit" + str(level)

if leakage == "True": #leakge models the system with an additional energy level. 
    level += 1

#Gate Workflow
tgate = gateGen(gateType,level)

#Speed limit Workflow
 #Needs to be change for different couplings, but do in the future
if gateType == "CNOT" or gateType == "CZ" or gateType == "CZ" or gateType == "CZ_0" or gateType == "CNOT_0":
    tmin = np.pi/4
elif gateType == "iSWAP":
    tmin = np.pi/2
elif gateType == "SWAP":
    tmin = 3*np.pi/4
elif gateType == "iTwoPhonon":
    tmin = np.pi/2
else:
    raise Exception("Incompatible target gate. Please see read me for example target gates (CNOT,CZ,iSWAP,etc.)")
#Coupling Workflow
if couplingType != "ContH":
    H0 = g*genCouplMat(couplingType,level)
else: 
    Evec = genEvec(anharmonicity,anharmonicity,staggering,g,level-1) # we divide by the coupling strength as everything we do is in units of g
    H0 = lambda t: HC(t,Evec,level-1)

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

#Warm Start Processing
warmStartBool = False
warmStartFinal = False
if warmStart != -1:
    warmStart = int(warmStart)
    warmStartBool = True 
    if warmStart == 1: warmStartFinal = True

#File creation
fname = quditType + "_" + gateType + "_" + couplingType + "_M" + str(segmentCount) + "_" + optimizer + "_" + ode
if leakage == "True": fname = fname + "_leakage"+ str(anharmonicity)
fname = fname + "_g" + str(g) + "_maxT" + str(maxTime)
#if anharmonicityType > 1: fname = fname + "_anharmType" + str(anharmonicityType)
if maxDriveStrength != -1: fname = fname + "_maxD" + str(maxDriveStrength)
if crossTalk != "False": fname = fname + "_CTh" + str(h) + "_stag" + str(staggering)
if ContPulse != "False": fname = fname + "_Cont" + str(ode) 
if couplingType == "ContH": fname = fname + "_ContH" + "_h" + str(h)
fname = fname + "_it" + str(iterationCount)
if minLeak == "True": fname = fname + "_leakMin" + str(alpha)
if warmStartBool: fname = fname + "_WS"

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

#Random Seed averaging 
#max_fidelity = 0
#seeds = np.random.randint(0,100,size=randomSeedCount)
# What is going on here? Check this out when reworking static seed code
if points != -1:
    hashseed = datetime.date.today().strftime("%Y%m%d") + str(seed)
    hash_object = hashlib.sha256(hashseed.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    rseed = hash_int % 1000

    times = np.linspace(minTime,maxTime,points)
    t = times[index]
else:
    rseed = 1
    t = maxTime

lbool = False
if leakage == "True":
    lbool = True

fname = os.path.join(fDir, fname + ".csv")
fWname = "Weights_t" + str(round(t,4)) + ".csv"
fWname = os.path.join(gDir, fWname)

fWnameRS = "False"
if warmStartBool:
    fWnameRS = "Weights" + "_RS" + str(rseed) + "_t" + str(round(t,4)) + ".csv"
    fWnameRS = os.path.join(gDir, fWnameRS)

[fidelity,W] = fidelity_ml(segmentCount,tgate,t*tmin,level,iterationCount,rseed,H0,drives,maxDriveStrength,lbool,minLeak,crossTalk,h,alpha,anharmonicity,staggering,ode,ContPulse,optimizer,fWnameRS,warmStartFinal)

flock = FileLock(fname + ".lock")
wlock = FileLock(fWname + ".lock")
if warmStartBool:
    wRSlock = FileLock(fWnameRS + ".lock")

def write():
    #Writing the fidelity
    out_arr = np.array([[fidelity,round(t,4)]]) #File output 
    with open(fname, 'a') as file:
        np.savetxt(file,out_arr,delimiter=",") #Fidelty writing
    if warmStartBool: #For Warm starts, we need to save the weights of each random seeds
        np.savetxt(fWnameRS,W,delimiter=",") #Weights writing
        wRSlock.release()
    else:
        np.savetxt(fWname,W,delimiter=",") #Weights writing
        wlock.release()
    flock.release()
    try:
        os.remove(fname + ".lock")
    except:
        pass
    if warmStartBool:
        try:
            os.remove(fWnameRS + ".lock")
        except:
            pass
    else:
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
if warmStartBool: #For Warm starts, we need to save the weights of each random seeds. Writing Weights 
    np.savetxt(fWnameRS,W,delimiter=",") #Weights writing

    if fidels["time"].isin([t]).any(): #if there is a fidelity from optimization convergence, then we check to write it
        tempFid = float(fidels[fidels["time"] == t]["fidelity"]) 
        if fidelity > tempFid: #if our fidelity is greater than the previous fidelity
            fIndex = fidels[fidels["time"] == t].index.to_numpy()[0] #What row our fidelity is at in the file
            fidels.iloc[fIndex,0] = fidelity
            fidels.to_csv(fname,index=False,header=False) #overwritting the previous file
            if warmStartFinal:
                np.savetxt(fWname,W,delimiter=",")
    else:
        write()
    if warmStartFinal:
        os.remove(fWnameRS)
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
else:
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