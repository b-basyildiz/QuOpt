#!/bin/bash
#SBATCH --job-name="QuOpt"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --time=24:00:00 

# Running respective job 
echo "Started"
 cd "$(dirname "$0")"
 # Ensure cfme package in repo is importable by adding CFME/src to PYTHONPATH
 REPO_ROOT="$(pwd)/.."
 export PYTHONPATH="$REPO_ROOT/CFME/src:$PYTHONPATH"

 # Forward all positional args to ControlFlow.py
 python ControlFlow.py "$@"
echo "Finished"
