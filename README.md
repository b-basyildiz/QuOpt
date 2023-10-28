# QuditML

<div align="center">
  <img src="Images/QCTRL2.png" alt="Image Description">
</div>


## Description
The speed of elementary quantum gates, particularly two-qubit gates, ultimately sets the limit on the speed at which quantum circuits can operate. In this work, we expand our computational system to a qutrit space and leverage our larger space to generate two-qubit gates at much faster speeds. Here, we show for specific two-qubit gates, we can theoretically achieve a 3x speed-up using experimentally feasible coupling regimes, and we prove that this speed-up is maximal for our regime. We accomplish this speed-up by utilizing a machine learning-inspired optimal control method, and our optimal control method also incorporates all error sources, leakage and cross-talk, from single-qudit drives. Using our method, we can nearly saturate our speed-up while accounting for non-instantaneous single-qudit drives and mitigating all controllable error sources. Our method generalizes for all weakly anharmonic systems, and thus our optimal control method is highly applicable for many experimental platforms and would significantly increase circuit depth. 

## Setup
1. Install a Python environment with version 3.7 and pip.
2. Make sure a version of Bash is installed on machine. 
3. Install GitHub repository on local machine. Ex(Linux/Unix): `git install https://github.com/b-basyildiz/QuditML.git`
4. Install required packages through `pip install requirements.txt` while in local environment.
5. All set to run! 

## Use
To run our Optimal Control Protocol, while in the local environment move to the Optimal_Control directory and open QOC.bash through `vim QOC.bash`, and this will look like
```vim
quditType="Qubit" #Qubit, Qutrit, 
gateType="CNOT" #CNOT, iSWAP, SWAP, iTwoPhonon

couplingType="XX" #XX, ZZ, XXX, Ashabb, AshhUnit, SpeedUp
maxDriveStrength=20 #natural number for capped max frequency, -1 for unlimited drive frequency

crossTalk="True" #models Cross Talk (CT), False for not CT, True for CT
contPulse="False" #whether or not to have continuous pulse shapes
leakage="False"

anharmonicity=5 #only used if larger than qubit system
staggering=15 # staggering of the two qudits in units of coupling strength, only relavent for Cross Talk

ode="SRK2" #RK2 or SRK2 
h=0.005 # step size for cross talk 

segmentCount=8
g=1
minTime=1.0
maxTime=1.2
points=1

randomSeedCount=-1
iterationCount=5000
optimizer="SGD"
```

## Examples
foo

## Credits 
foo

## References
foo
