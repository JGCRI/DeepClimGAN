#!/bin/zsh

#SBATCH -n 8
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
program="DeepClimGAN/Trainer.py"
data_dir="/pic/projects/GCAM/DeepClimGAN-input/MIROC5_Tensors_norm"
gen_data_dir="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/MIROC5_Gen/"
real_data_dir="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/MIROC5_Real/"
norms_dir="/pic/projects/GCAM/DeepClimGAN-input/MIROC5_Tensors/norms/"
batch_size=32
num_epoch=5
report_loss=500
nodelist=`scontrol show hostnames $SLURM_JOB_NODELIST`
NODERANK=0
report_loss=500
save_gen_data_update=5000
n_data_to_save=320
pretrain=0
save_model_dir="/pic/projects/GCAM/DeepClimGAN-input/data_for_gan_test/saved_model/"



date
echo "starting master process on node $MASTER"	
srun -n 1 -N 1 python -m torch.distributed.launch --nnodes $SLURM_JOB_NUM_NODES --nproc_per_node 2 --node_rank=$NODERANK --master_addr=$MASTER	--master_port=$PORT $program --num_epoch $num_epoch --batch_size $batch_size --data_dir=$data_dir --report_loss=$report_loss --gen_data_dir=$gen_data_dir --real_data_dir=$real_data_dir --n_data_to_save=$n_data_to_save --save_gen_data_update=$save_gen_data_update --norms_dir=$norms_dir --pretrain=$pretrain --save_model_dir=$save_model_dir&

sleep 2

while read -r node; do
	if [[ $node != $MASTER ]]; then
		NODERANK=$(( NODERANK + 1 ))
		echo "launching process on node $node (noderank=$NODERANK)"	
		srun -n 1 -N 1 python -m torch.distributed.launch --nnodes $SLURM_JOB_NUM_NODES --nproc_per_node 2 --node_rank=$NODERANK --master_addr=$MASTER --master_port=$PORT  $program --num_epoch $num_epoch --batch_size $batch_size --data_dir $data_dir --report_loss=$report_loss --gen_data_dir=$gen_data_dir --real_data_dir=$real_data_dir --n_data_to_save=$n_data_to_save --save_gen_data_update=$save_gen_data_update --norms_dir=$norms_dir --pretrain=$pretrain --save_model_dir=$save_model_dir&
	fi
done <<< "$nodelist"

wait

date
