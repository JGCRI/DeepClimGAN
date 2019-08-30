import sys
from Utils import sort_files_by_size
import argparse
import torch
import torch.distributed as dist
import os
from Constants import scenarios, realizations, clmt_vars
from Normalizer import Normalizer
from Utils import snake_data_partition
import logging


def main(partition, nrm, file_dir, logfile, rank):	
	comm_size = dist.get_world_size()
	
	tas_tsr, tasmin_tsr, tasmax_tsr, rhsmax_tsr, rhsmin_tsr, rhs_tsr = None, None, None, None, None, None
	#tas_tsrs, tasmin_tsrs, tasmax_tsrs, rhsmax_tsrs, rhsmin_tsrs, rhs_tsrs = [[] for i in range(6)]
	tas_glob_sum, tasmin_glob_sum, tasmax_glob_sum, rhs_glob_sum, rhsmin_glob_sum, rhsmax_glob_sum = 0, 0, 0, 0, 0 ,0	
	tsr_glob_size = 0
	clmt_var_keys = list(clmt_vars.keys())
	
	for file in partition:
		ts_name = os.path.join(file_dir, file)
		tsr = torch.load(ts_name).detach()
		logfile.write(f'{ts_name}\n')
		logfile.write(f'tsr shape: {tsr.shape}\n')
		
		tas_tsr = tsr[clmt_var_keys.index('tas')]
		tasmin_tsr = tsr[clmt_var_keys.index('tasmin')]
		tasmax_tsr = tsr[clmt_var_keys.index('tasmax')]
		rhsmin_tsr = tsr[clmt_var_keys.index('rhsmin')]
		rhsmax_tsr = tsr[clmt_var_keys.index('rhsmax')]
		rhs_tsr = tsr[clmt_var_keys.index('rhs')]
		
		tas_loc_sum, tsr_loc_size = torch.sum(tas_tsr), tas_tsr.shape[-1]
		tasmin_loc_sum = torch.sum(tasmin_tsr)
		tasmax_loc_sum = torch.sum(tasmax_tsr)
		rhsmin_loc_sum = torch.sum(rhsmin_tsr)
		rhsmax_loc_sum = torch.sum(rhsmax_tsr)
		rhs_loc_sum = torch.sum(rhs_tsr)

		tas_glob_sum += tas_loc_sum
		tasmin_glob_sum += tasmin_loc_sum
		tasmax_glob_sum += tasmax_loc_sum
		rhs_glob_sum += rhs_loc_sum
		rhsmin_glob_sum += rhsmin_loc_sum
		rhsmax_glob_sum += rhsmax_loc_sum

		tsr_glob_size += tsr_loc_size

	logfile.write(f'Sums before broadcasting')
	logfile.write(f'Size of tas_glob_sum: {tas_glob_sum}\n')
	logfile.write(f'Size of tasmin_glob_sum: {tasmin_glob_sum}\n')
	logfile.write(f'Size of tasmax_glob_sum: {tasmax_glob_sum}\n')
	logfile.write(f'Size of rhs_glob_sum: {rhs_glob_sum}\n')
	logfile.write(f'Size of rhsmin_glob_sum: {rhsmin_glob_sum}\n')
	logfile.write(f'Size of rhsmax_glob_sum: {rhsmax_glob_sum}\n')


	
	logfile.flush()	
	tsr_size = torch.tensor(tsr_glob_size)
	
	logfile.write(f'Size of tensors for partition: {tsr_size}\n')
	logfile.flush()
	#compute the total size of tensors and distribute the value for all
	all_reduce_dist([tsr_size])
	logfile.write(f'Total size of tensors: {tsr_size}\n')
	logfile.flush()
	
	all_reduce_dist([tas_glob_sum, tasmin_glob_sum, tasmax_glob_sum, rhs_glob_sum, rhsmin_glob_sum, rhsmax_glob_sum])	
	
	logfile.write(f'Sums after broadcasting')
	logfile.write(f'Size of tas_glob_sum: {tas_glob_sum}\n')
	logfile.write(f'Size of tasmin_glob_sum: {tasmin_glob_sum}\n')
	logfile.write(f'Size of tasmax_glob_sum: {tasmax_glob_sum}\n')
	logfile.write(f'Size of rhs_glob_sum: {rhs_glob_sum}\n')
	logfile.write(f'Size of rhsmin_glob_sum: {rhsmin_glob_sum}\n')
	logfile.write(f'Size of rhsmax_glob_sum: {rhsmax_glob_sum}\n')
	
	#compute the means
	map_size = 128 * 256
	tsr_size = tsr_size * map_size
	
	tas_mean = torch.tensor(tas_glob_sum / tsr_size)
	tasmax_mean = torch.tensor(tasmax_glob_sum / tsr_size)
	tasmin_mean =  torch.tensor(tasmin_glob_sum / tsr_size)
	rhs_mean = torch.tensor(rhs_glob_sum / tsr_size)
	rhsmin_mean = torch.tensor(rhsmin_glob_sum / tsr_size)
	rhsmax_mean = torch.tensor(rhsmax_glob_sum / tsr_size)
	
	#compute variance
	tas_sq_diff = torch.sum((tas_tsr - tas_mean) ** 2)
	tasmin_sq_diff = torch.sum((tasmin_tsr - tasmin_mean)** 2)
	tasmax_sq_diff = torch.sum((tasmax_tsr - tasmax_mean) ** 2)
	rhs_sq_diff = torch.sum((rhs_tsr - rhs_mean) ** 2)
	rhsmin_sq_diff = torch.sum((rhsmin_tsr - rhsmin_mean) ** 2)
	rhsmax_sq_diff = torch.sum((rhsmax_tsr - rhsmax_mean) ** 2)


	#reduce sum of squared differences among all the nodes
	all_reduce_dist([tas_sq_diff, tasmin_sq_diff, tasmax_sq_diff, rhs_sq_diff, rhsmin_sq_diff, rhsmax_sq_diff])	

	#compute std
	tas_std = torch.sqrt(tas_sq_diff / tsr_size)
	tasmin_std = torch.sqrt(tasmin_sq_diff / tsr_size)
	tasmax_std = torch.sqrt(tasmax_sq_diff / tsr_size)
	rhs_std = torch.sqrt(rhs_sq_diff / tsr_size)
	rhsmin_std = torch.sqrt(rhsmin_sq_diff / tsr_size)
	rhsmax_std = torch.sqrt(rhsmax_sq_diff / tsr_size)

	nrm.clmt_stats['tas'] = [tas_mean, tas_std]
	nrm.clmt_stats['tasmin'] = [tasmin_mean, tasmin_std]
	nrm.clmt_stats['tasmax'] = [tasmax_mean, tasmax_std]
	nrm.clmt_stats['rhs'] = [rhs_mean, rhs_std]
	nrm.clmt_stats['rhsmin'] = [rhsmin_mean, rhsmin_std]
	nrm.clmt_stats['rhsmax'] = [rhsmax_mean, rhsmax_std]
	

	if rank == 0:
		#save means from one process
		torch.save(tas_mean, os.path.join(file_dir,'tas_mean'))
		torch.save(tasmax_mean, os.path.join(file_dir, 'tasmax_mean'))
		torch.save(tasmin_mean, os.path.join(file_dir, 'tasmin_mean'))
		torch.save(rhs_mean, os.path.join(file_dir, 'rhs_mean'))
		torch.save(rhsmin_mean, os.path.join(file_dir, 'rhsmin_mean'))
		torch.save(rhsmax_mean, os.path.join(file_dir, 'rhsmax_mean'))


		#save std from one process
		torch.save(tas_std, os.path.join(file_dir,'tas_std'))
		torch.save(tasmax_std, os.path.join(file_dir, 'tasmax_std'))
		torch.save(tasmin_std, os.path.join(file_dir, 'tasmin_std'))
		torch.save(rhs_std, os.path.join(file_dir, 'rhs_std'))
		torch.save(rhsmin_std, os.path.join(file_dir, 'rhsmin_std'))
		torch.save(rhsmax_std, os.path.join(file_dir, 'rhsmax_std'))
		

	#normalize and save all tensors
	for file in partition:
		ts_name = os.path.join(file_dir, file)
		tsr = torch.load(ts_name)
		norm_tsr = nrm.normalize(tsr)
		norm_ts_name = os.path.join(file_dir, 'norm_' + file)
		torch.save(norm_tsr, norm_ts_name)
	
	logfile.write(f'Finished normalization')


def all_reduce_dist(sums):
	for sum in sums:
		dist.all_reduce(sum, op=dist.ReduceOp.SUM)


if __name__ == '__main__':
	file_dir = '/pic/projects/GCAM/DeepClimGAN-input/MIROC5_Tensors/'		
	sorted_files = sort_files_by_size(file_dir)
	
	nrm = Normalizer()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--local_rank', type=int)
	args = parser.parse_args()
	rank = args.local_rank
	
	dist.init_process_group('gloo', 'env://')

	print(f'localrank: {rank}	host: {os.uname()[1]}')

	#create mapping for nodes to process the files
	comm_size = dist.get_world_size()
	
	partition = snake_data_partition(sorted_files, comm_size)
	
	rank = dist.get_rank()
	
	logfilename = f'log-{rank}.txt'
	with open(logfilename, 'w') as logfile:
		main(partition[rank], nrm, file_dir, logfile, rank)
	
