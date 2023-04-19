#Imports 
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from fidelity_subQutrit import Qutrit_ML

#Qutrit Subsystem Graph Generation 

#Gate and ferromagnetic variables
CNOT_qutrit = torch.tensor([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0],
[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]],dtype=torch.cdouble)
J = torch.tensor([[1,1],[1,1]])

#Iteration variables 
Fidelities = []
Times = []
points = np.linspace(0,0.2*np.pi/4,20)
points = np.append(points, np.linspace(0.3*np.pi/4,np.pi/4,10))
iteration_count = 10000

Pulse_file = "CNOT_Qutrit_Experimental_NoSecondDrive2"

#File Creation 
try: #Making Data Folder
    os.makedirs("Data")
except:
    pass
try: #Making Figures Folder
    os.makedirs("Figures")
except:
    pass
try: #Making Pulse_Sequences Folder
    os.makedirs("Pulse_Sequences")
except:
    pass
try:
    os.makedirs("Pulse_Sequences/"+Pulse_file)
except:
    shutil.rmtree("Pulse_Sequences/"+Pulse_file)
    os.makedirs("Pulse_Sequences/"+Pulse_file)
    

#Generating Points
for t in points:
    Times.append(t/(np.pi/4))
    Fidelities.append(1 - Qutrit_ML(J,[1,1],8,CNOT_qutrit,t,iteration_count,Pulse_file))

#Saving Fidelities
np.savetxt(os.path.join(os.getcwd(),"Data",Pulse_file+".csv"),Fidelities,delimiter=",")

#plotting 
plt.plot(Times,Fidelities,'o-')
plt.xlabel("T/Tmin")
plt.ylabel("Fidelity")
plt.title(str(Pulse_file))
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle='dotted', linewidth='0.5')
plt.minorticks_on()
plt.savefig(os.path.join(os.getcwd(),"Figures",Pulse_file+".pdf"), format="pdf")

