#!bin/bash 

for i in "Qutrit_nop2" "Qutrit_nop"
do
   #for j in "CNOT" "iSWAP" "SWAP"
   for j in "CNOT" 
   do
        for k in {1..20}
        do
            python pythonControl.py $i $j $k
        done
    done
done