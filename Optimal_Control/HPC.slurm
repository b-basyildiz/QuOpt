#!/bin/bash
#SBATCH --job-name="QuOpt"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --time=08:00:00 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bbasyildiz@mines.edu


#Load Anaconda Module 
module load anaconda

#Activate Conda Environment, python 3.7.9
conda activate OptCntrl

# Running respective job 
python ControlFlow.py $1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 $12 $13 $14 $15 $16 $17 $18 $19 $20 $21 
