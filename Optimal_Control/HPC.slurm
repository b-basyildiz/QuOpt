#!/bin/bash
#SBATCH --job-name="QuOpt"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --time=24:00:00 

# Running respective job 
echo "Started"
python ControlFlow.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24}
echo "Finished"
