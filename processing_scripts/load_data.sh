#!/bin/sh
#SBATCH -n 1
#SBATCH -t 4-0
#SBATCH -A dlclim
#SBATCH --gres=gpu:2

date

module load python/anaconda3
program="DeepClimGAN/DataPreprocessor.py"
python3 ./$program /pic/projects/GCAM/DeepClimGAN-input/MIROC5/ /pic/projects/GCAM/DeepClimGAN-input/MIROC5_Tensors/ 1950 2009

date
