#!/bin/bash
gateType="CZ" #CNOT, iSWAP, SWAP, iTwoPhonon, CNOT_0, CZ_0
level=3 #Size of computational space. For qubit gates (CNOT), d = 2. For qutrit gates, d = 3, etc.

couplingType="capacitiveCoupMin" #XX, ZZ, XXX, capacitiveCoup, SpeedUp, ContH
maxDriveStrength=40 #natural number for capped max frequency, -1 for unlimited drive frequency

crossTalk="True" #models Cross Talk (CT), False for not CT, True for CT
contPulse="False" #whether or not to have continuous pulse shapes
leakage="True"
minimizeLeakage="False" #whether or not to penalize higher energy states 

anharmonicity=14 #only used if larger than qubit system
staggering=17 # staggering of the two qudits in units of coupling strength, only relavent for Cross Talk

ode="CFME4" #RK2, SRK2, CFME4
h=0.01 # step size for cross talk 
alpha=0.5 # Tuning parameter for leakage minimization 

segmentCount=1
g=1
minTime=0.1
maxTime=0.2
points=1

randomSeedCount=1
iterationCount=1
optimizer="SGD"

HPC="False" #HPC or local bool. If local, make 'False'
WarmStart=-1 #Submitting chained jobs that use each others weights as a 'warm start.' -1 if no warm start.


set -euo pipefail

# Ensure script runs from its directory and prepare output folder
cd "$(dirname "$0")"
mkdir -p slurm_out

# Loop for the specified number of iterations
if [ "$randomSeedCount" -eq -1 ]; then
    # Fixed-seed mode
    for ((i=0; i<points; i++)); do
        if [ "$HPC" = "True" ]; then
            sbatch HPC.slurm "$gateType" "$level" "$couplingType" "$segmentCount" "$g" "$anharmonicity" "$crossTalk" "$staggering" "$ode" "$h" "$alpha" "$contPulse" "$leakage" "$minimizeLeakage" "$maxDriveStrength" "$minTime" "$maxTime" "$points" "$iterationCount" "$optimizer" "$i" "$randomSeedCount" "$WarmStart"
        elif [ "$HPC" = "False" ]; then
            python ControlFlow.py "$gateType" "$level" "$couplingType" "$segmentCount" "$g" "$anharmonicity" "$crossTalk" "$staggering" "$ode" "$h" "$alpha" "$contPulse" "$leakage" "$minimizeLeakage" "$maxDriveStrength" "$minTime" "$maxTime" "$points" "$iterationCount" "$optimizer" "$i" "$randomSeedCount" "$WarmStart"
        else
            echo "Incorrect computing location. Either HPC or local machine."
            exit 1
        fi
    done
else
    # Random-seed mode
    for ((i=0; i<points; i++)); do
        for ((j=0; j<randomSeedCount; j++)); do
            if [ "$HPC" = "True" ]; then
                sbatch HPC.slurm "$gateType" "$level" "$couplingType" "$segmentCount" "$g" "$anharmonicity" "$crossTalk" "$staggering" "$ode" "$h" "$alpha" "$contPulse" "$leakage" "$minimizeLeakage" "$maxDriveStrength" "$minTime" "$maxTime" "$points" "$iterationCount" "$optimizer" "$i" "$j" "$WarmStart"
            elif [ "$HPC" = "False" ]; then
                python ControlFlow.py "$gateType" "$level" "$couplingType" "$segmentCount" "$g" "$anharmonicity" "$crossTalk" "$staggering" "$ode" "$h" "$alpha" "$contPulse" "$leakage" "$minimizeLeakage" "$maxDriveStrength" "$minTime" "$maxTime" "$points" "$iterationCount" "$optimizer" "$i" "$j" "$WarmStart"
            else
                echo "Incorrect computing location. Either HPC or local machine."
                exit 1
            fi
        done
    done
fi

#wait 
#DESC: 
#  Drive Types: 
#    - all: both |0> <-> |1> & |1> <-> |2> drives
#    - qtd (qutrit drives): only |1> <-> |2> drives
#    - yd (y drives): only y drive from |1> <-> |2> drive
#    - twoPhon(on)All: both |0> <-> |1> & |0> <-> |2> drive
#    - twoPhon(on)All: only |0> <-> |2> drives
#    - leakage[int]: |0> <-> |1> drives and |1> <-> |2> drives with the same pulses, int is the anharmonicity value
#  Coupling Types:
#    - XX: XX coupling in either the qubit or qutrit spaces
#    - ZZ: ZZ coupling in either the qubit or qutrit spaces
#    - XXX: XX coupling from both (|0> <-> |1> & |1> <-> |2>)
#    - Ashhab: capactive coupling seen in Ashhab's qutrit paper
#    - AshhUnit: Ashhab couplings Hamiltonian with unit couplings
#    - Analy(itical): derived analytical couplings protocol 
#  
#  Gates:
#    - iTwoPhonon:two phonon transition (|00> -> |11>) with a phase of i.
#
#  PAST DESC (not supported anymore):
#    Coupling Types: 
#      - AnalyNeg: analytical protocol with a permutation of negatives on the non-zero entries
