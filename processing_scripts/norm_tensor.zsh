#!/bin/zsh

#SBATCH -A dlclim
#SBATCH --ntasks-per-node=1

module load gcc
module load python/anaconda3


program="DeepClimGAN/norm.py"

date


tid=$SLURM_ARRAY_TASK_ID

python3 ./$program $tid

date
