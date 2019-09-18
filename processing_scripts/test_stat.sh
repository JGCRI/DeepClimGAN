#!/bin/sh
#SBATCH -n 1
#SBATCH -t 4-0
#SBATCH -A dlclim
#SBATCH --gres=gpu:2


module load python/anaconda3
module load gcc/8.1.0
module load netcdf/4.4.1.1

date

tid=$SLURM_ARRAY_TASK_ID

program="DeepClimGAN/get_statistics.py"

save_gen_data_update=100
gen_data_dir="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/MIROC5_Gen/"
real_data_dir="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/MIROC5_Real/"
gen_csv_dir="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/gen_csv/"
real_csv_dir="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/real_csv/"
stats_dir="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/stats/"
norms_dir="/pic/projects/GCAM/DeepClimGAN-input/MIROC5_Tensors/norms/"
exp_id=29


python3 ./$program --gen_data_dir=$gen_data_dir --real_data_dir=$real_data_dir --gen_csv_dir=$gen_csv_dir --real_csv_dir=$real_csv_dir --stats_dir=$stats_dir  --exp_id=$exp_id --norms_dir=$norms_dir

date
