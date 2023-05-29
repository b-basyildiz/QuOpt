#!bin/bash 

for i in "Qubit" "Qubit_gRoot2"
do
   for j in "iSWAP" 
   do
        for k in {1..20}
        do
            python pythonControl.py $i $j $k
        done
    done
done