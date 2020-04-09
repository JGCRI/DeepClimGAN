#!/bin/zsh

#SBATCH -n 1
#SBATCH -p fat
#SBATCH -t 4-0
#SBATCH -A dlclim
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1


module load gcc/8.1.0
module load python/anaconda3.2019.3


date

echo "nodes: $SLURM_JOB_NODELIST"	

env | grep SLURM


MASTER=`hostname -s`
PORT=24601
program="DeepClimGAN/TrainerAdversar.py"
data_dir="/pic/projects/GCAM/DeepClimGAN-input/MIROC5_Tensors_norm"
gen_data_dir="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/MIROC5_Gen/"
real_data_dir="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/MIROC5_Real/"
norms_dir="/pic/projects/GCAM/DeepClimGAN-input/MIROC5_Tensors/norms/"
batch_size=32
num_epoch=3
report_loss=500
nodelist=`scontrol show hostnames $SLURM_JOB_NODELIST`
NODERANK=0
report_loss=500
save_gen_data_update=5000
n_data_to_save=320
save_model_dir="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/saved_model/"
z_shape=256
pretrained_model="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/saved_model/exp_110/"
n_generate_for_one_z=0
dir_to_save_z_realizations="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/z_realizations/"
num_smoothing_conv_layers=0
last_layer_size=1
n_days=32
label_smoothing=0
train_G_with_context=1
train_D_with_lowres_context=0
train_D_with_highres_context=1
start_lr=0.05
end_lr=0.0999
add_mean_loss=1
partition_start=2

date
python $program --num_epoch $num_epoch --batch_size $batch_size --data_dir=$data_dir --report_loss=$report_loss --gen_data_dir=$gen_data_dir --real_data_dir=$real_data_dir --n_data_to_save=$n_data_to_save --save_gen_data_update=$save_gen_data_update --norms_dir=$norms_dir --save_model_dir=$save_model_dir --z_shape=$z_shape --pretrained_model=$pretrained_model --n_generate_for_one_z=$n_generate_for_one_z --dir_to_save_z_realizations=$dir_to_save_z_realizations --num_smoothing_conv_layers=$num_smoothing_conv_layers --last_layer_size=$last_layer_size --n_days=$n_days --label_smoothing=$label_smoothing --train_G_with_context=$train_G_with_context --train_D_with_highres_context=$train_D_with_highres_context --train_D_with_lowres_context=$train_D_with_lowres_context --start_lr=$start_lr --end_lr=$end_lr --add_mean_loss=$add_mean_loss --partition_start=$partition_start&

sleep 2

wait

date
