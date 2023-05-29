#!bin/bash 

for i in "all" "qutrits" "y_qutrit"
do
    for j in {1..20}
    do
        python workflow_iSWAP.py $i $j
    done
done