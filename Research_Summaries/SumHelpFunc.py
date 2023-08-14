'''
DESC: This file helperFuncs.py provides helper functions for the research functions
INPUTS: N/A
OUTPUTS: N/A
AUTHOR: Bora Basyildiz
'''

#Importing Packages
import matplotlib.pyplot as plt
import os
from numpy import array,concatenate,ones,kron,sqrt,zeros
import pandas as pd 

def genPlot(df,l=[""],m="o",msize=5,c="#1f77b4"):
    '''
    DESC: Generates a plot for a given dataframe input. \n

    PARAMS: 
        - df: Pandas Dataframe with columns [fidelity, time]
        - l: legend (default is none)
        - m: marker type (default is circle)
        - msize: marker size (default is 5)
        - c: graph color (default is blue) \n

    RETURNS: Generates formatted MatPlotLib plot \n

    AUTHOR: Bora Basyildiz
    '''
    plt.rcParams["mathtext.fontset"] = 'cm' #Allows matplotlib plots to have latex eqs/symbols
    plt.plot(df["time"],df["fidelity"],marker=m,markersize=msize,color=c)
    plt.xlabel('$T/T_{min}$',fontsize=16)
    plt.ylabel('$F$',fontsize=16)
    plt.grid(which='major', linestyle='-', linewidth='0.5')
    plt.grid(which='minor', linestyle='dotted', linewidth='0.5')
    plt.minorticks_on()
    plt.legend(l)

def saveFig(date,title):
    '''
    DESC: Saves a generated figure to the correct folder.  \n

    PARAMS: 
        - date: Reserch Summary Date 
        - titile: title of figure \n

    RETURNS: Saves figure \n

    AUTHOR: Bora Basyildiz 
    '''
    mainDir = "../Figures/Summary_" + str(date) + "/"
    try: #All files are stored under their gateType
        os.makedirs(mainDir)
    except:
        pass 
    plt.savefig(os.path.join(os.getcwd(),mainDir,str(title) + ".pdf"), format="pdf")

def readData(fname):
    '''
    DESC: Processes Wendian Data from Filename \n

    PARAMS:
        - fname: file name for data (includes path to file) \n

    RETURNS: Pandas Dataframe of sorted data \n

    AUTHOR: Bora Basyildiz
    '''
    df = pd.read_csv(fname,names=["fidelity","time"])
    df.sort_values(by=["time"],inplace=True)
    return df

def qubitSubspace(maxLevel): 
    '''
    DESC: Generates the state needed to find the occupation of the qubit subspace. \n

    PARAMS: 
        - maxLevel: Maximum energy levels of the system. \n

    RETURNS: Wavefunction that is essentially hadamards in the qubit system. \n

    AUTHOR: Bora Basyildiz
    '''
    return 1/2*concatenate((array([1,1]),array((maxLevel-1)*[0]),array([1,1]),array(((maxLevel+1) ** 2 - maxLevel - 1)*[0])))


def stateProj(eLevel,maxLevel,output=False):
    '''
    DESC: Generates the state needed to find the occupation of a higher energy level. \n

    PARAMS: 
        - eLevel: Energy level of interest.
        - maxLevel: Maximum energy levels of the system.
        - output: optional printing statement. \n

    RETURNS: Wavefunction that spans energy level subspace. \n

    AUTHOR: Bora Basyildiz
    '''
    psi = ones(maxLevel+1)
    psi[eLevel] = 2
    psi = kron(psi,psi)
    psi = (psi != 1).astype(int)
    for i in range(len(psi)): # This is to ensure no population of higher energy states
        if i > eLevel*(maxLevel+1) + eLevel: psi[i] = 0
    psi = psi/sqrt(sum(psi))
    if output == True:
        for i in range(maxLevel+1):
            for j in range(maxLevel+1):
                print("|" + str(i) + str(j) + ">: " + str(psi[(maxLevel+1)*i + j]))
    return psi

def genDrive(d, dTrans,type):
    '''
    DESC: Creates drives based on the dimension and transition wanted. \n

    PARAMS: 
        - d: energy level of the system 
        - dTrans: the transition wanted in the drive (between either dTrans and dTrans - 1) or a two Phonon transition
        - type: Type of drive wanted. Ex: X,Y,Z, or TPX, or TPY. TP stands for two phonon transition. \n

    OUTPUT: d x d numpy matrix with selected transition \n

    AUTHOR: Bora Basyildiz
    '''
    if d < 2:
        raise Exception("The energy levels of our system must be greater than 2. Your input has " + str(d) + " energy levels.")
    if dTrans == 0 or dTrans > d - 1:
        raise Exception("The X gate must be between energy levels withn our system. Right now your transition is in between |" + str(dTrans-1)  + "> -> |" + str(dTrans) + ">.\n Number of energy levels is " + str(d) + ".")
    type = type.lower()
    if type != "x" and type != "y" and type != "tpx" and type != "tpy":
        raise Exception("The drive must be an X, Y, or two Phonon transition (input X, Y, TPX, or TPY).")

    cIndex = 1
    if type[:2] == "tp": 
        cIndex = 2
        type = type[2:] 

    drive = zeros((d,d),dtype=complex)
    val = 0
    if type == "x": val = 1
    else: val = -1j
    drive[dTrans-cIndex,dTrans] = val
    return drive + drive.conj().T