import sys
from Utils import sort_files_by_size
import argparse
import torch
import torch.distributed as dist
import os
import numpy as np
from Constants import scenarios, realization, clmt_vars
from Normalizer import Normalizer

def main(sorted_files, node_storage, nrm):
	
	rank = dist.get_rank()
	comm_size = dist.get_world_size()
	
	#TODO:
	#size = end = 0
	#while size < node_storage_files:
	#	size += os.path.getsize(sorted_files[pt])	
	#	end += 1
	#
	#files = sorted_files[start:end]


	tas_tsr, tasmin_tsr, tasmax_tsr, rhsmax_tsr, rhsmin_tsr, rhs_tsr = None
	tas_tsrs, tasmin_tsrs, tasmax_tsrs, rhsmax_tsrs, rhsmin_tsrs, rhs_tsrs = [[] for i in range(6)]

	for file in files:
		tsr = torch.load(file)
		if 'tas' in file:
			tas_tsrs.append(tsr[clmt_vars['tas']])
		elif 'tasmin' in file:
			tasmin_tsrs.append(tsr[clmt_vars['tasmin']])
		elif 'tasmax' in file:
			tasmax_tsrs.append(tsr[clmt_vars['tasmax']])
		elif 'rhsmin' in file:
			rhsmin_tsrs.append(tsr[clmt_vars['rhsmin']])
		elif 'rhsmax' in file:
			rhsmax_tsrs.append(tsr[clmt_vars['rhsmax']])
		elif 'rhs' in file:
			rhs_tsrs.append(tsr[clmt_vars['rhs']])
	

	#stack all variables
	tas_tsr = torch.stack(tas_tsrs, dim=0)
	tasmin_tsr = torch.stack(tasmin_tsrs, dim=0)
	tasmax_tsr = torch.stack(tasmax_tsrs, dim=0)
	rhsmax_tsr = torch.stack(rhsmax_tsrs, dim=0)
	rhsmin_tsr = torch.stack(rhsmin_tsrs, dim=)
	rhs_tsr = torch.stack(rhs_tsrs, dim=0)
	tsr_size = tas_tsr.shape[0]
		
	#compute the total size of tensors 
	all_reduce_dist([tsr_sz])
	
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
	tas_sq_diff = tsr[clmt_vars['tas']] - tas_mean
	tasmin_sq_diff = tsr[clmt_vars['tasmin']] - tasmin_mean
	tasmax_sq_diff = tsr[clmt_vars['tasmax']] - tasmax_mean
	rhs_sq_diff = tsr[clmt_vars['rhs']] - rhs_mean
	rhsmin_sq_diff =tsr[clmt_vars['rhsmin']] - rhsmin_mean
	rhsmax_sq_diff = tsr[clmt_vars'rhsmax']] - rhsmax_mean


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
	for file in files_per_node:
		tsr = torch.load(file)
		norm_tsr = nrm.normalize(tsr)
		torch.save(norm_tsr, 'norm_' + 'file' + '.pt')



def all_reduce_dist(tensors):
	for t in tensors:
		dist.all_reduce(t, op=dist.reduce_op.SUM)


if __name__ == '__main__':
	file_dir = sys.argv[1]		
	sorted_files = sort_files_by_size(file_dir)
	
	nrm = Normalizer()

	
	parser = argparse.ArgumentParser()
	parser.add_argument('--local_rank', type=int)
	args = parser.parse_args()

	rank = args.local_rank
	node_size = args.node_size
	
	print(f'localrank: {rank}	host: {os.uname()[1]}')

	
	dist.init_process_group('gloo', 'env://')
	main(sorted_files, node_storage, nrm)
	
	
