#!/bin/bash
quditType="Qutrit" #Qubit, Qutrit
gateType="CNOT" #CNOT, iSWAP, SWAP
couplingType="Diagonal" #XX, Ashabb, iSWAP_Protocol, CNOT_Protocol, Speed_Up, Diagonal
SegmentNum=8
g=1
drives_type="all" #all, qtd, yd, dipoleAll, dipoleQtd

for i in {1..20}
    do
        python fullControl.py $quditType $gateType $couplingType $SegmentNum $drives_type $g $i
    done