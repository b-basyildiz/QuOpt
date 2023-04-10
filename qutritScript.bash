#!/bin/bash
#SBATCH --job-name="Qutrit_ML"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --time=01:00:00 <- what

cd $SCRATCH
ls > myfiles
srun hostname
