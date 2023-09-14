#!/bin/bash
quditType="Qubit" #Qubit, Qutrit, 
gateType="CNOT" #CNOT, iSWAP, SWAP, iTwoPhonon

couplingType="XX" #XX, ZZ, XXX, Ashabb, AshhUnit, SpeedUp
segmentCount=2
g=1

drivesType="all" #all, qtd, yd, twoPhonAll, twoPhonQtd, leakage
anharmonicity=5 #only used if larger than qubit system

crossTalk="False" #models Cross Talk (CT), False for not CT, True for CT
staggering=15 # staggering of the two qudits in units of coupling strength, only relavent for Cross Talk

ode="RK2" #RK2 or SRK2 
h=0.001 # step size for cross talk 

ContPulse="False"
maxDriveStrength=20 #natural number for capped max frequency, -1 for unlimited drive frequency

maxTime=1
points=1

randomSeedCount=1
iterationCount=1000

# Loop for the specified number of iterations
for ((i=1; i<=points; i++))
do
    #sbatch HPC.slurm $quditType $gateType $couplingType $segmentCount $g $drivesType $anharmonicity $crossTalk $staggering $ode $h $ContPulse $maxDriveStrength $maxTime $points $randomSeedCount $iterationCount $i
    python ControlFlow.py $quditType $gateType $couplingType $segmentCount $g $drivesType $anharmonicity $crossTalk $staggering $ode $h $ContPulse $maxDriveStrength $maxTime $points $randomSeedCount $iterationCount $i
    : 'echo -e "\nQudit Type: "$quditType"\nGate Type: "$gateType"\nCoupling Type: "$couplingType"\nSegment Number:"$segmentNum"
Drive Type:"$drivesType"\nAnharmonicity: "$anharmonicity"\nCrossTalk"$crossTalk"\nCoupling Strength: "$g"
Random Seed Count: "$randomSeedCount"\nNumber of Points: "$points"\nML Iteration Count: "$iterationCount "
Max Drive Strength: "$maxDriveStrength"\nMaximum Time: "$maxTime"\nPoint Number: "$i "\n" '
done

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
