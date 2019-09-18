files = ["rcp26_r2i1p1.pt", "rcp26_r3i1p1.pt", "rcp45_r1i1p1.pt", "rcp45_r2i1p1.pt", "rcp60_r1i1p1.pt"]


from Normalizer import Normalizer
import os
import torch
import sys


def main():
	nrm = Normalizer()
	file_idx = int(sys.argv[1])
		

	file_dir = '/pic/projects/GCAM/DeepClimGAN-input/MIROC5_Tensors/'
	

	#means
	tas_mean = torch.load(os.path.join(file_dir, 'tas_mean.pt'))
	tasmin_mean = torch.load(os.path.join(file_dir, 'tasmin_mean.pt'))
	tasmax_mean = torch.load(os.path.join(file_dir, 'tasmax_mean.pt'))
	rhs_mean = torch.load(os.path.join(file_dir, 'rhs_mean.pt'))
	rhsmin_mean = torch.load(os.path.join(file_dir, 'rhsmin_mean.pt'))
	rhsmax_mean = torch.load(os.path.join(file_dir, 'rhsmax_mean.pt'))


	#sts
	tas_std = torch.load(os.path.join(file_dir, 'tas_std.pt'))
	tasmin_std = torch.load(os.path.join(file_dir, 'tasmin_std.pt'))
	tasmax_std = torch.load(os.path.join(file_dir, 'tasmax_std.pt'))
	rhs_std = torch.load(os.path.join(file_dir, 'rhs_std.pt'))
	rhsmin_std = torch.load(os.path.join(file_dir, 'rhsmin_std.pt'))
	rhsmax_std = torch.load(os.path.join(file_dir, 'rhsmax_std.pt'))

	nrm.clmt_stats['tas'] = [tas_mean, tas_std]
	nrm.clmt_stats['tasmin'] = [tasmin_mean, tasmin_std]
	nrm.clmt_stats['tasmax'] = [tasmax_mean, tasmax_std]
	nrm.clmt_stats['rhs'] = [rhs_mean, rhs_std]
	nrm.clmt_stats['rhsmin'] = [rhsmin_mean, rhsmin_std]
	nrm.clmt_stats['rhsmax'] = [rhsmax_mean, rhsmax_std]

	file = files[file_idx]

	ts_name = os.path.join(file_dir, file)
	tsr = torch.load(ts_name)
	norm_tsr = nrm.normalize(tsr)
	
	norm_ts_name = os.path.join(file_dir, 'norm_' + file)
	torch.save(norm_tsr, norm_ts_name)
	




main()
