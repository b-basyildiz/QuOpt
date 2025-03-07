#!/bin/bash
quditType="Qutrit" #Qubit, Qutrit, 
gateType="CNOT" #CNOT, iSWAP, SWAP, iTwoPhonon, CNOT_0, CZ_0
d=2 #Size of computational space. For qubit gates (CNOT), d = 2. For qutrit gates (CZZ), d = 3

couplingType="capacitiveCoup" #XX, ZZ, XXX, capacitiveCoup, SpeedUp, ContH
maxDriveStrength=40 #natural number for capped max frequency, -1 for unlimited drive frequency

crossTalk="False" #models Cross Talk (CT), False for not CT, True for CT
contPulse="False" #whether or not to have continuous pulse shapes
leakage="True"
minimizeLeakage="False" #whether or not to penalize higher energy states 

anharmonicity=10 #only used if larger than qubit system
staggering=15 # staggering of the two qudits in units of coupling strength, only relavent for Cross Talk

ode="SRK2" #RK2 or SRK2 
h=0.0001 # step size for cross talk 
alpha=0.5 # Tuning parameter for leakage minimization 

segmentCount=1
g=1
minTime=0.1
maxTime=0.1
points=1

randomSeedCount=1
iterationCount=1
optimizer="SGD"

HPC="False" #HPC or local bool. If local, make 'False'
WarmStart=-1 #Submitting chained jobs that use each others weights as a 'warm start.' -1 if no warm start.


# Loop for the specified number of iterations
if [ $((randomSeedCount)) -eq -1 ]; then #This is for a fixed seed
    for ((i=0; i<points; i++))
    do
        if [ "$HPC" = "True" ]; then 
            if [ $WarmStart -eq -1 ]; then
                sbatch HPC.slurm $quditType $gateType $d $couplingType $segmentCount $g $anharmonicity $crossTalk $staggering $ode $h $alpha $contPulse $leakage $minimizeLeakage $maxDriveStrength $minTime $maxTime $points $iterationCount $optimizer $i $randomSeedCount $WarmStart
            else
                for ((k=WarmStart; k>0; k--))
                do
                    if [ $k -eq ${WarmStart} ]; then
                        OUTPUT=$(sbatch HPC.slurm $quditType $gateType $d $couplingType $segmentCount $g $anharmonicity $crossTalk $staggering $ode $h $alpha $contPulse $leakage $minimizeLeakage $maxDriveStrength $minTime $maxTime $points $iterationCount $optimizer $i $randomSeedCount $WarmStart)
                    else
                        OUTPUT=$(sbatch --dependency=afterok:${JOB_ID} HPC.slurm $quditType $gateType $d $couplingType $segmentCount $g $anharmonicity $crossTalk $staggering $ode $h $alpha $contPulse $leakage $minimizeLeakage $maxDriveStrength $minTime $maxTime $points $iterationCount $optimizer $i $randomSeedCount $k)
                    fi
                    JOB_ID=$(echo ${OUTPUT} | awk '{print $4}')
                    echo "${OUTPUT}"	
                done
            fi
        elif [ "$HPC" = "False" ]; then
            python ControlFlow.py $quditType $gateType $d $couplingType $segmentCount $g $anharmonicity $crossTalk $staggering $ode $h $alpha $contPulse $leakage $minimizeLeakage $maxDriveStrength $minTime $maxTime $points $iterationCount $optimizer $i $randomSeedCount $WarmStart
        else 
            echo "Incorrect computing location. Either HPC or local machine."
            exit
        fi 
    done
else #This is for random seeds
    for ((i=0; i<points; i++))
    do
        for ((j=0; j<randomSeedCount; j++))
        do
            if [ "$HPC" = "True" ]; then 
                if [ $WarmStart -eq -1 ]; then
                    sbatch HPC.slurm $quditType $gateType $d $couplingType $segmentCount $g $anharmonicity $crossTalk $staggering $ode $h $alpha $contPulse $leakage $minimizeLeakage $maxDriveStrength $minTime $maxTime $points $iterationCount $optimizer $i $j $WarmStart
                else
                    for ((k=WarmStart; k>0; k--))
                    do
                        if [ $k -eq ${WarmStart} ]; then
                            OUTPUT=$(sbatch HPC.slurm $quditType $gateType $d $couplingType $segmentCount $g $anharmonicity $crossTalk $staggering $ode $h $alpha $contPulse $leakage $minimizeLeakage $maxDriveStrength $minTime $maxTime $points $iterationCount $optimizer $i $j $k)
                        else
                            OUTPUT=$(sbatch --dependency=afterok:${JOB_ID} HPC.slurm $quditType $gateType $couplingType $segmentCount $g $anharmonicity $crossTalk $staggering $ode $h $alpha $contPulse $leakage $minimizeLeakage $maxDriveStrength $minTime $maxTime $points $iterationCount $optimizer $i $j $k)
                        fi
                        JOB_ID=$(echo ${OUTPUT} | awk '{print $4}')
                        echo "${OUTPUT}"	
                    done
                fi
            elif [ "$HPC" = "False" ]; then
                python ControlFlow.py $quditType $gateType $d $couplingType $segmentCount $g $anharmonicity $crossTalk $staggering $ode $h $alpha $contPulse $leakage $minimizeLeakage $maxDriveStrength $minTime $maxTime $points $iterationCount $optimizer $i $j $WarmStart
            else 
            echo "Incorrect computing location. Either HPC or local machine."
            exit
        fi
        #     : 'echo -e "\nQudit Type: "$quditType"\nGate Type: "$gateType"\nCoupling Type: "$couplingType"\nSegment Number:"$segmentNum"
        # Drive Type:"$drivesType"\nAnharmonicity: "$anharmonicity"\nCrossTalk"$crossTalk"\nCoupling Strength: "$g"
        # Random Seed Count: "$randomSeedCount"\nNumber of Points: "$points"\nML Iteration Count: "$iterationCount "
        # Max Drive Strength: "$maxDriveStrength"\nMaximum Time: "$maxTime"\nPoint Number: "$i "\n" '
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
