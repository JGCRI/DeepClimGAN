import sys
from Utils import sort_files_by_size
import argparse
import torch
import torch.distributed as dist
import os
import numpy as np
from Constants import scenarios, realizations, clmt_vars
from Normalizer import Normalizer
from Utils import snake_data_partition
import logging

def main(partition, nrm, file_dir, logfile):
	
	
	comm_size = dist.get_world_size()
	
	tas_tsr = tasmin_tsr = tasmax_tsr = rhsmax_tsr = rhsmin_tsr = rhs_tsr = None
	tas_tsrs, tasmin_tsrs, tasmax_tsrs, rhsmax_tsrs, rhsmin_tsrs, rhs_tsrs = [[] for i in range(6)]
		
	clmt_var_keys = list(clmt_vars.keys())

	
	for file in partition:
		ts_name = os.path.join(file_dir, file)
		tsr = torch.load(ts_name)
		logfile.write(f'{ts_name}\n')
	
	logfile.flush()

	'''	
		tas_tsrs.append(tsr[clmt_var_keys.index('tas')])
		tasmin_tsrs.append(tsr[clmt_var_keys.index('tasmin')])
		tasmax_tsrs.append(tsr[clmt_var_keys.index('tasmax')])
		rhsmin_tsrs.append(tsr[clmt_var_keys.index('rhsmin')])
		rhsmax_tsrs.append(tsr[clmt_var_keys.index('rhsmax')])
		rhs_tsrs.append(tsr[clmt_var_keys.index('rhs')])
	
	
	
	#stack all variables
	tas_tsr = torch.stack(tas_tsrs, dim=0)
	tasmin_tsr = torch.stack(tasmin_tsrs, dim=0)
	tasmax_tsr = torch.stack(tasmax_tsrs, dim=0)
	rhsmax_tsr = torch.stack(rhsmax_tsrs, dim=0)
	rhsmin_tsr = torch.stack(rhsmin_tsrs, dim=0)
	rhs_tsr = torch.stack(rhs_tsrs, dim=0)
	tsr_size = tas_tsr.shape[0]
	
	
	with open (ofname, 'w') as of:
		of.write(f'{tsr_size}\n')
		of.flush()


	#compute the total size of tensors and distribute the value for all
	all_reduce_dist([tsr_size])
	
	with open(ofname, 'w') as of:
		of.write(f'{tsr_size}')
		of.flush()

	# reduce sums among all the nodes
	all_reduce_dist([tas_tsr, tasmin_tsr, tasmax_tsr, rhs_tsr, rhsmin_tsr, rhsmax_tsr])
	
	
	#compute the means
	tas_mean = tas_tsr[0] / tsr_sz
	tasmax_mean = tasmax_tsr[0] / tsr_sz
	tasmin_mean =  tasmin_tsr[0] / tsr_sz
	rhs_mean = rhs_tsr[0] / tsr_sz
	rhsmin_mean = rhsmin_tsr[0] / tsr_sz
	rhsmax_mean = rhsmax_tsr[0] / tsr_sz
	

	#compute variance
	tas_sq_diff = tsr[clmt_var_keys.index('tas')] - tas_mean
	tasmin_sq_diff = tsr[clmt_var_keys.index('tasmin')] - tasmin_mean
	tasmax_sq_diff = tsr[clmt_var_keys.index('tasmax')] - tasmax_mean
	rhs_sq_diff = tsr[clmt_var_keys.index('rhs')] - rhs_mean
	rhsmin_sq_diff =tsr[clmt_var_keys.index('rhsmin')] - rhsmin_mean
	rhsmax_sq_diff = tsr[clmt_var_keys.index('rhsmax')] - rhsmax_mean


	#reduce sum of squared differences among all the nodes
	all_reduce_dist([tas_sq_diff, tasmin_sq_diff, tasmax_sq_diff, rhs_sq_diff, rhsmin_sq_diff, rhsmax_sq_diff])	

	#compute std
	tas_std = np.sqrt(tas_sq_diff[0] / tsr_sz)
	tasmin_std = np.sqrt(tasmin_sq_diff[0] / tsr_sz)
	tasmax_std = np.sqrt(tasmax_sq_diff[0] / tsr_sz)
	rhs_std = np.sqrt(rhs_sq_diff[0] / tsr_sz)
	rhsmin_std = np.sqrt(rhsmin_sq_diff[0] / tsr_sz)
	rhsmax_std = np.sqrt(rhsmax_sq_diff[0] / tsr_sz)



	#normalize and save all tensors all tensors
	for file in files_per_proc:
		ts_name = os.path.join(file_dir, file)
		tsr = torch.load(ts_name)
		norm_tsr = nrm.normalize(tsr)
		norm_ts_name = os.path.join(file_dir, 'norm_' + file + '.pt')
		torch.save(norm_tsr, norm_ts_name)

	'''

def all_reduce_dist(tensors):
	for t in tensors:
		dist.all_reduce(t, op=dist.reduce_op.SUM)


if __name__ == '__main__':
	file_dir = '/pic/projects/GCAM/DeepClimGAN-input/MIROC5_Tensors/'		
	sorted_files = sort_files_by_size(file_dir)
	
	nrm = Normalizer()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--local_rank', type=int)
	args = parser.parse_args()
	n_proc_per_node = 2
	rank = args.local_rank
	n_nodes = 2
	#running on the fat node
	node_size = 360  * 1073741824
	empty_space = node_size * 0.02  #use only 98 % of the node capacity
	dist.init_process_group('gloo', 'env://')

	print(f'localrank: {rank}	host: {os.uname()[1]}')

	#create mapping for nodes to process the files
	comm_size = dist.get_world_size()
	partition = snake_data_partition(sorted_files, comm_size)
	
	print(partition)
	rank = dist.get_rank()
	
	logfilename = f'log-{rank}.txt'
	with open(logfilename, 'w') as logfile:
		main(partition[rank], nrm, file_dir, logfile)
	
