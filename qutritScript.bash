#!/bin/bash
#SBATCH --job-name="Qutrit_ML"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --time=00:01:00 


#Load Python Module
module load apps/python3/2020.02

#Activate Conda Environment
source activate py7

cd $SCRATCH
ls > myfiles
srun python fidelity_subQutrit.py 
