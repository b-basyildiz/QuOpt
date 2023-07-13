#!/bin/bash
quditType="Qutrit" #Qubit, Qutrit
gateType="iTwoPhonon" #CNOT, iSWAP, SWAP, iTwoPhonon
couplingType="Analy" #XX, Ashabb, AshhUnit, Analy
segmentNum=16
g=1
drivesType="dipoleAll" #all, qtd(only 1<->2), yd(only 1<->2 y), dipoleAll(0<->1 & 0<->2), dipoleQtd(0<->2)
randomSeedCount=50
points=5
iterationCount=1
maxDriveStrength=-1 #natural number for capped max frequency, -1 for unlimited drive frequency
maxTime=1

#this is for random seeding of AnalyNeg coupling
if [ "$couplingType" == "AnalyNeg" ]; then
    rn=$((0 + RANDOM % 24))
    couplingType=$couplingType$rn
fi


# Loop for the specified number of iterations
for ((i=1; i<=points; i++))
do
    #sbatch fullSub.slurm $quditType $gateType $couplingType $segmentNum $drivesType $g $randomSeedCount $points $iterationCount $maxDriveStrength $maxTime $i
    python fullControl.py $quditType $gateType $couplingType $segmentNum $drivesType $g $randomSeedCount $points $iterationCount $maxDriveStrength $maxTime $i
    #echo -e "\nQudit Type: "$quditType"\nGate Type: "$gateType"\nCoupling Type: "$couplingType"\nSegment Number:"$segmentNum"\nDrive Type:"$drivesType"\nCoupling Strength: "$g"\nRandom Seed Count: "$randomSeedCount"\nNumber of Points: "$points"\nML Iteration Count: "$iterationCount"\nMax Drive Strength: "$maxDriveStrength"\nMaximum Time: "$maxTime"\nPoint Number: "$i "\n"
done

#DESC: 
#  Coupling Types:
#    - XX is just an XX coupling in either the qubit or qutrit spaces
#    - Ashhab is the capactive coupling seen in Ashhab's qutrit paper
#    - AshhUnit is the Ashhab couplings Hamiltonian with unit couplings
#    - Analy is the derived analytical couplings protocol 
#    - AnalyNeg is the analyticla protocol with a permutation of negatives on the non-zero entries
