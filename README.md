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
gateType="CNOT" #CNOT, iSWAP, SWAP

couplingType="XX" #XX, ZZ, SpeedUp
maxDriveStrength=20 #natural number for capped max frequency, -1 for uncapped drive frequency

crossTalk="True" #models Cross Talk (CT), "False" for not CT, "True" for CT
contPulse="False" #models continuous pulse shapes with sin^2(x) pulses, similar input to CT
leakage="False" #models leakage to a higher energy state, similar input to CT

anharmonicity=5 #detuning of energy states (in units of the coupling strength). Anharmonicity is the same for both qudits. 
staggering=15 #staggering between of the two qudits (in units of coupling strength)

ode="SRK2" #RK2 (Runge-Kutta Second Order) or SRK2 (Symplectic Runge-Kutta Second Order)
h=0.005 # Step size for ODE solver

segmentCount=8 #segment count
g=1 #coupling strength
minTime=1.0 #minimum time to generate gate in point spread
maxTime=1.2 #maximum time to generate gate in point spread
points=1 #number of points in spread

randomSeedCount=-1 #amount of random seeds to run in parallel (-1 for a fixed seed)
iterationCount=5000 #number of iterations for optimizer 
optimizer="SGD" #Optimizer used to tune parameters (SGD: Stochastic Gradient Descent, ADAM, and others (see helpFuncs.py))
```

## Examples
foo

## Credits 
foo

## References
foo
