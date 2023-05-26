#!bin/bash 

for i in "Qubit" "Qutrit"
do
    if [ $i == "Qubit" ]
    then
        for k in {1..20}
        do
            python StateTransControl.py $i 'qubit' $k
            #echo $i $k
        done
    else 
        for j in "num_op" "normalized" "ones" 
            do
                for k in {1..20}
                do
                    python StateTransControl.py $i $j $k
                    #echo $i $j $k
                done
            done
    fi
done
