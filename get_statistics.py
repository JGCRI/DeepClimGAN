"""
get_statistics.py is used to get different kinds of statistics
for the generated and ground truth data.

"""


from Constants import clmt_vars, grid_cells_for_stat
from scipy import stats
import sys
import numpy as np
import argparse
import os
import torch
import logging
import csv
from Normalizer import Normalizer
import netCDF4 as n
N_CELLS = len(grid_cells_for_stat)


def process_tensors(data_path, exp_id, nrm, cmp_z_realizations):

	clmt_var_keys = list(clmt_vars.keys())
	tas_tsrs, tasmin_tsrs, tasmax_tsrs = [], [], []
	rhs_tsrs, rhsmin_tsrs , rhsmax_tsrs = [], [], []
	pr_tsrs = []
	#tensor dim: 7 x 128 x 256 x 32
	for filename in os.listdir(data_path):
		ts_name = os.path.join(data_path, filename)
		tsr = torch.load(ts_name)
		tsr = tsr.detach().cpu()
		#print(tsr.shape)		
		#Denormalize tensor
		if cmp_z_realizations:
			tsr = tsr[0]
		tsr = nrm.denormalize(tsr)
		pr_tsrs.append(tsr[clmt_var_keys.index('pr')])
		#tas_tsrs.append(tsr[clmt_var_keys.index('tas')])
		#tasmin_tsrs.append(tsr[clmt_var_keys.index('tasmin')])
		#tasmax_tsrs.append(tsr[clmt_var_keys.index('tasmax')])
		#rhsmin_tsrs.append(tsr[clmt_var_keys.index('rhsmin')])
		#rhsmax_tsrs.append(tsr[clmt_var_keys.index('rhsmax')])
		#rhs_tsrs.append(tsr[clmt_var_keys.index('rhs')])

	#return (pr_tsrs, tas_tsrs, tasmin_tsrs, tasmax_tsrs, rhs_tsrs, rhsmin_tsrs, rhsmax_tsrs)
	#return pr_tsrs, tas_tsrs, tasmin_tsrs, tasmax_tsrs
	return pr_tsrs

def merge_tensors(tsrs):

	#pr_tsrs, tas_tsrs, tasmin_tsrs, tasmax_tsrs, rhs_tsrs, rhsmin_tsrs, rhsmax_tsrs = tsrs
	#pr_tsrs, tas_tsrs, tasmin_tsrs, tasmax_tsrs = tsrs
	pr_tsrs = tsrs
	pr_tsr = torch.cat(pr_tsrs, dim=2)
	#tas_tsr = torch.cat(tas_tsrs, dim=2)
	#tasmin_tsr = torch.cat(tasmin_tsrs, dim=2)
	#tasmax_tsr = torch.cat(tasmax_tsrs, dim=2)
	#rhs_tsr = torch.cat(rhs_tsrs, dim=2)
	#rhsmin_tsr = torch.cat(rhsmin_tsrs, dim=2)
	#rhsmax_tsr = torch.cat(rhsmax_tsrs, dim=2)
	N_DAYS = pr_tsr.shape[-1]

	#rhs_tsr = rhsmin_tsr = rhsmax_tsr = None
	#return pr_tsr, tas_tsr, tasmin_tsr, tasmax_tsr, rhs_tsr, rhsmin_tsr, rhsmax_tsr
	#return pr_tsr, tas_tsr, tasmin_tsr, tasmax_tsr
	return pr_tsr

def get_tas_stat(tas_tsr):
	"""
	Output:
	 mean
	 sd/variance
	 p-value for Shapiro-Wilk
	"""
	
	N_DAYS = tas_tsr.shape[-1]
	dic = [[0 for _ in range(3)] for _ in range(N_CELLS)]
	tas_tsr = np.asarray(tas_tsr)
	
	for i, (cell, val) in enumerate(grid_cells_for_stat.items()):
		lat, lon = val
		tsr = tas_tsr[lat, lon]
		mean = np.mean(tsr)
		std = np.std(tsr)
		tsr = np.asarray(tsr)
		test_stat, p_value = stats.shapiro(tsr)
		dic[i][:] = [mean, std, p_value]
	
	return dic	



def get_p_stat_for_cells(p_tsr):
	"""
	Return 5 stat vals per each cell grid
	"""

	dic = [[0 for _ in range(5)] for _ in range(N_CELLS)]
	
	for i, (cell, val) in enumerate(grid_cells_for_stat.items()):
		lat, lon = val
		tsr = p_tsr[lat, lon]
		dic[i][:] = get_p_stat(tsr)
	
	return dic
						

def get_p_stat(pr):
	"""
	Get stat for precipitation
	"""	

	N_DAYS = pr.shape[-1]
	pr = np.asarray(pr)
	pr_mean = np.mean(pr)
	n_zero_vals = len(pr[pr == 0.0])
	zero_fraq = n_zero_vals / N_DAYS
	non_zero = pr[pr != 0.0]
	non_zero_mean = np.mean(non_zero)
	non_zero_std = np.std(non_zero)

	return [pr_mean, zero_fraq, non_zero_mean, non_zero_std]



def get_tas_min_max_stat_for_cells(tas, tasmin, tasmax):
	"""
	Returns 5 stat values per each cell
	"""
	dic = [[0 for _ in range(5)] for _ in range(N_CELLS)]
	
	for i, (cell, val) in enumerate(grid_cells_for_stat.items()):
		lat, lon = val
		tas_tsr = tas[lat, lon]
		tasmin_tsr = tasmin[lat, lon]
		tasmax_tsr = tasmax[lat, lon]

		dic[i][:] = get_tas_min_max_stat(tas_tsr, tasmin_tsr, tasmax_tsr)

	return dic


def get_tas_min_max_stat(tas, tasmin, tasmax):
	"""
	return 5 stat values
	"""
	

	N_DAYS = tas.shape[-1]
	tas_arr  = np.asarray(tas)
	tasmin = np.asarray(tasmin)
	tasmax = np.asarray(tasmax)
	
	tasmin_mean = np.mean(tasmin)
	tasmax_mean = np.mean(tasmax)
	
	tasmin_std = np.std(tasmin)
	tasmax_std = np.std(tasmax)
		
	tas_fltr = tas_arr[(tas_arr >= tasmin) & (tas_arr <= tasmax)]
	
	tas_is_in_range_fraq = len(tas_fltr) / N_DAYS
	
	return [tasmin_mean, tasmax_mean, tasmin_std, tasmax_std, tas_is_in_range_fraq]


def get_rhs_stat_for_cells(rhs):
	"""
	Returns 4 stat values
	"""
	
	dic = [[0 for _ in range(4)] for _ in range(N_CELLS)]

	for i, (cell, val) in enumerate(grid_cells_for_stat.items()):
		lat, lon = val
		rhs_tsr = rhs[lat, lon]
		dic[i][:] = get_rhs_stat(rhs_tsr)

	return dic

def get_rhs_stat(rhs):
	"""
	Returns 4 stat values for rhs
	"""
	rhs = np.asarray(rhs)	
	
	rhs_min = np.min(rhs)
	rhs_max = np.max(rhs)
	rhs_mean = np.mean(rhs)
	rhs_std = np.std(rhs)	
	
	return [rhs_min, rhs_max, rhs_mean, rhs_std]
	

def get_rhs_min_max_stat_for_cells(rhs, rhsmin, rhsmax):
	"""
	Returns stat values
	"""
	dic = [[0 for _ in range(5)] for _ in range(N_CELLS)]
	
	for i, (cell, val) in enumerate(grid_cells_for_stat.items()):
		lat, lon = val
		rhs_tsr = rhs[lat, lon]
		rhsmin_tsr = rhsmin[lat, lon]
		rhsmax_tsr = rhsmax[lat, lon]
		dic[i][:] = get_rhs_min_max_stat(rhs_tsr, rhsmin_tsr, rhsmax_tsr)

	return dic

def get_rhs_min_max_stat(rhs, rhsmin, rhsmax):
	
	"""
	Return 5 stat vars
	"""

	N_DAYS = rhs.shape[-1]
	
	rhs_arr = np.asarray(rhs)
	rhsmin = np.asarray(rhsmin)
	rhsmax = np.asarray(rhsmax)
	
	rhsmin_mean = np.mean(rhsmin)
	rhsmax_mean = np.mean(rhsmax)
	
	rhsmin_std = np.std(rhsmin)
	rhsmax_std = np.std(rhsmax)

	rhs_fltr = rhs_arr[(rhs_arr >= rhsmin) & (rhs_arr <= rhsmax)]
	
	rhs_is_in_range_fraq = len(rhs_fltr) / N_DAYS
	
	return [rhsmin_mean, rhsmax_mean, rhsmin_std, rhsmax_std, rhs_is_in_range_fraq]



def get_dewpoint_stat_for_cells(rhs, tas):
	
	dic = [[0 for _ in range(5)] for _ in range(N_CELLS)]

	for i, (cell, val) in enumerate(grid_cells_for_stat.items()):
		lat, lon = val
		rhs_tsr = rhs[lat, lon]
		tas_tsr = tas[lat, lon]
		dic[i][:] = get_dewpoint_stat(rhs_tsr, tas_tsr)

	return dic


def get_dewpoint_stat(rhs, tas):
	"""
	Calculate statistics for dewpoint
	"""
	
	dp = calc_dewpoint(rhs, tas)	
	dp_min = np.min(dp)
	dp_max = np.max(dp)
	dp_mean = np.mean(dp)
	dp_std = np.std(dp)

	return [dp_min, dp_max, dp_mean, dp_std]

def calc_dewpoint(rhs, tas):
	
	alpha = float(6.112)
	beta = float(17.62)
	lambda_v = 243.12
	
	rhs = np.asarray(rhs)
	tas = np.asarray(tas)
	num = np.log(rhs / 100) + (beta * tas) / (lambda_v + tas)
	dp = lambda_v * num / (beta - num)
	return dp




def write_stats_to_csv(stats, fname, exp_id, type):
	"""
	precip, tas, tasmin/max, rhs, rhsmin/max
	"""
	fname = os.path.join(fname, "exp_" + str(exp_id) + "/" + type + "/stats.csv")
	with open(fname, mode='w+', newline='') as of:
		c_writer = csv.writer(of, delimiter=',')
		for i in range(len(stats)):
			c_writer.writerow(stats[i])
			c_writer.writerow('\n')


def write_to_csv_for_time_series(tsrs, path, prefix):
	"""
	tsrs: tensor of all cliamte variable, size: N_vars x 128 x 256 x 10240, where
		10240 - 320 samples of months in 32 days
	"""


	#print(tsrs.shape)
	#tsrs = list(tsrs)
	
	#print(tsrs[0].shape)
	#print(len(tsrs))
	ts_ctr = 0
	tsrs = [tsrs] #input variable tsrs is just a precipiation
	for tsr in tsrs:
		ts_name = os.path.join(path, prefix + "_" + str(ts_ctr) + ".csv")
		with open(ts_name, 'w+') as of:
			cv_writer = csv.writer(of, delimiter=',')
			for i, (cell, val) in enumerate(grid_cells_for_stat.items()):
				lat, lon = val
				tensor = tsr[lat, lon]
				arr = np.asarray(tensor)
				cv_writer.writerow(arr)					
				
		ts_ctr += 1


def check_pr(pr_real):
	pr = np.asarray(pr_real)
	N_days = pr.shape[-1]
	zero_fraq = len(pr[pr == 0]) / N_days * 128 * 256
	#print("neg fraq {}".format(zero_fraq))
	pr_norm = np.log(1 + pr)
	negatives = pr_norm[pr_norm < 0]
	print("Negative values in pr after log norm {}".format(len(negatives)))
	
	relu_pr_norm = abs(pr_norm)
	denorm_pr = np.exp(relu_pr_norm)- 1
	print("pr ground truth \n")
	print(pr[0][1][:100])
	print("\n")

	print("pr after normalization -> relu ->  denormalized \n")
	print(denorm_pr[0][1][:100])
	print("\n")

def check_rhs(rhs_real, mean, std):
	rhs = np.asarray(rhs_real)
	rhs_norm = (rhs - mean) / std	
	
	#apply sigmoid
	rhs_sigm = sigmoid(rhs_norm)
	
	#denormalize values
	rhs_denorm = rhs_sigm * std + mean
	
	print("rhs ground truth\n")
	print(rhs[0][0][:100])
	print("\n")

	print("rhs after normalization -> sigmoid -> denormalization")
	print(rhs_denorm[0][0][:100])
	print("\n")


def sigmoid(x):
	return 1 / (1 + np.exp(-x))	



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gen_data_dir', type=str)
	parser.add_argument('--real_data_dir', type=str)
	parser.add_argument('--gen_csv_dir', type=str)
	parser.add_argument('--real_csv_path', type=str)
	parser.add_argument('--stats_dir', type=str)
	parser.add_argument('--exp_id', type=int)
	parser.add_argument('--real_csv_dir', type=str)
	parser.add_argument('--norms_dir', type=str)
	parser.add_argument('--save_for_panoply', type=int)
	parser.add_argument('--n_to_save_for_panoply',type=int)
	parser.add_argument('--for_panoply_dir', type=str)
	parser.add_argument('--cmp_z_realizations', type=int)
	parser.add_argument('--cmp_z_realizations_dir', type=str)
	
	args = parser.parse_args()
	real_data_dir = args.real_data_dir
	gen_data_dir = args.gen_data_dir
	real_csv_dir = args.real_csv_dir
	gen_csv_dir = args.gen_csv_dir
	stats_dir = args.stats_dir
	exp_id = args.exp_id
	real_csv_dir = args.real_csv_dir	
	norms_dir = args.norms_dir
	save_for_panoply = args.save_for_panoply
	n_to_save_for_panoply = args.n_to_save_for_panoply #how many months
	for_panoply_dir = args.for_panoply_dir
	cmp_z_realizations = args.cmp_z_realizations
	cmp_z_realizations_dir = args.cmp_z_realizations_dir	


	real_csv_dir = real_csv_dir + "exp_" + str(exp_id) + "/"
	gen_csv_dir = gen_csv_dir + "exp_" + str(exp_id) + "/"
	

	
	nrm = Normalizer()
	nrm.load_means_and_stds(norms_dir)
		
	if cmp_z_realizations:
		gen_data_dir = cmp_z_realizations_dir + "exp_" + str(exp_id) + "/gen/" 
		real_data_dir = cmp_z_realizations_dir + "exp_" + str(exp_id) + "/real/"
	else:
		gen_data_dir = gen_data_dir + "exp_" + str(exp_id) + "/"
		real_data_dir = real_data_dir + "exp_" + str(exp_id) + "/"

	tsrs = process_tensors(gen_data_dir, exp_id, nrm, cmp_z_realizations)
	#pr_gen, tas_gen, tasmin_gen, tasmax_gen = merge_tensors(tsrs)	
	pr_gen = merge_tensors(tsrs)

	real_tsrs = process_tensors(real_data_dir, exp_id,  nrm, cmp_z_realizations)
	#pr_real, tas_real, tasmin_real, tasmax_real = merge_tensors(real_tsrs)
	pr_real = merge_tensors(real_tsrs)
	#pr_months = np.asarray(pr_real[:,:, 0:32])
	#pr_mean = np.mean(pr_months, axis=-1)
	#for_panoply_dir = for_panoply_dir + "exp_" + str(exp_id) + "/gen/"
	#np.save(for_panoply_dir + 'pr_mean', pr_mean)
	#return

	one_month = 32 #one month = 32 days
	n_maps = one_month * n_to_save_for_panoply
	print("n_maps num{}".format(n_maps))
	panoply_dir = for_panoply_dir	




	if save_for_panoply:
		#save gen
		if cmp_z_realizations:
			for_panoply_dir = panoply_dir + "exp_" + str(exp_id) + "/z_realizations/gen/" 
		else:
			for_panoply_dir = panoply_dir + "exp_" + str(exp_id) + "/gen/"
		pr_months = np.asarray(pr_gen[:,:, 0:n_maps])
	
		#tas_months = np.asarray(tas_gen[:,:,0:n_maps])
		#tasmin_months = np.asarray(tasmin_gen[:,:,0:n_maps])
		#tasmax_months = np.asarray(tasmax_gen[:,:,0:n_maps])
		#rhs_months = np.asarray(rhs_gen[:,:,0:n_maps])
		#rhsmin_months = np.asarray(rhsmin_gen[:,:,0:n_maps])
		#rhsmax_months = np.asarray(rhsmax_gen[:,:,0:n_maps])
		np.save(for_panoply_dir + 'pr', pr_months)
		#np.save(for_panoply_dir + 'tas', tas_months)
		#np.save(for_panoply_dir + 'tasmin', tasmin_months)
		#np.save(for_panoply_dir + 'tasmax', tasmax_months)
		#np.save(for_panoply_dir + 'rhs', rhs_months)
		#np.save(for_panoply_dir + 'rhsmin', rhsmin_months)
		#np.save(for_panoply_dir + 'rhsmax', rhsmax_months)
 
	
		#save real
		if cmp_z_realizations:
			for_panoply_dir	= panoply_dir +	"exp_" + str(exp_id) + "/z_realizations/real/"
		else:
			for_panoply_dir = panoply_dir + "exp_" + str(exp_id) + "/real/"

		pr_months = np.asarray(pr_real[:,:, 0:n_maps])
		#tas_months = np.asarray(tas_real[:,:,0:n_maps])
		#tasmin_months = np.asarray(tasmin_real[:,:,0:n_maps])
		#tasmax_months = np.asarray(tasmax_real[:,:,0:n_maps])
		#rhs_months = np.asarray(rhs_real[:,:,0:n_maps])
		#rhsmin_months = np.asarray(rhsmin_real[:,:,0:n_maps])
		#rhsmax_months = np.asarray(rhsmax_real[:,:,0:n_maps])
		np.save(for_panoply_dir + 'pr_real', pr_months)
		#np.save(for_panoply_dir + 'tas_real', tas_months)
		#np.save(for_panoply_dir + 'tasmin_real', tasmin_months)
		#np.save(for_panoply_dir + 'tasmax_real', tasmax_months)
		#np.save(for_panoply_dir + 'rhs_real', rhs_months)
		#np.save(for_panoply_dir + 'rhsmin_real', rhsmin_months)
		#np.save(for_panoply_dir + 'rhsmax_real', rhsmax_months)	
		


	#Generated data stats
	p_stat = get_p_stat_for_cells(pr_gen)
	#tas_stats = get_tas_stat(tas_gen)
	#tas_min_max_stat = get_tas_min_max_stat_for_cells(tas_gen, tasmin_gen, tasmax_gen)
	#rhs_stats = get_rhs_stat_for_cells(rhs_gen)
	#rhsmin_max = get_rhs_min_max_stat_for_cells(rhs_gen, rhsmin_gen, rhsmax_gen)
	#rhs_stats = rhsmin_max = None
	#dewpoint = get_dewpoint_stat_for_cells(rhs_gen, tas_gen)
	#dewpoint = None
	#gen_stats = [p_stat, tas_stats, tas_min_max_stat]
	gen_stats = [p_stat]
	write_stats_to_csv(gen_stats, stats_dir, exp_id, "gen")
	
	#save ptime series for precip from 2 processes
	
	#print(tsrs[0].shape)
	precip_gen = tsrs[0]
	#precip_gen = tsrs[0][:2]
	#print(tsrs[0].shape)
	#print(precip_gen.shape)
	#print(tsrs[0].shape)
	#tas_gen = tsrs[1][:2]
	#tasmin_gen = tsrs[2][:2]
	#tasmax_gen = tsrs[3][:2]
	#rhs_gen = tsrs[4][:2]
	#rhsmin_gen = tsrs[5][:2]
	#rhsmax_gen = tsrs[6][:2]	
	
	
	write_to_csv_for_time_series(precip_gen, gen_csv_dir, "pr")
	#write_to_csv_for_time_series(tas_gen, gen_csv_dir, "tas")
	#write_to_csv_for_time_series(tasmin_gen, gen_csv_dir, "tasmin")
	#write_to_csv_for_time_series(tasmax_gen, gen_csv_dir, "tasmax")
	#write_to_csv_for_time_series(rhs_gen, gen_csv_dir, "rhs")
	#write_to_csv_for_time_series(rhsmin_gen, gen_csv_dir, "rhsmin")
	#write_to_csv_for_time_series(rhsmax_gen, gen_csv_dir, "rhsmax")

	#Real data stats
	p_stat_real = get_p_stat_for_cells(pr_real)
	#tas_stat_real = get_tas_stat(tas_real)
	#tas_min_max_stat_real = get_tas_min_max_stat_for_cells(tas_real, tasmin_real, tasmax_real)
	#rhs_stat_real = get_rhs_stat_for_cells(rhs_real)
	#rhsmin_max_real = get_rhs_min_max_stat_for_cells(rhs_real, rhsmin_real, rhsmax_real)
	#rhs_stat_real = rhsmin_max_real = None
	#dewpoint_real = get_dewpoint_stat_for_cells(rhs_real, tas_real)
	#dewpoint_real = None
	real_stats = [p_stat_real]
	#real_stats = [p_stat_real, tas_stat_real, tas_min_max_stat_real]	
	write_stats_to_csv(real_stats, stats_dir, exp_id, "real")

	precip_real = real_tsrs[0]
	#precip_real = real_tsrs[0][0:2]
	#tas_real = real_tsrs[1][0:2]
	#tasmin_real = real_tsrs[2][0:2]
	#tasmax_real = real_tsrs[3][0:2]
	#rhs_real = real_tsrs[4][0:2]
	#rhsmin_real = real_tsrs[5][0:2]
	#rhsmax_real = real_tsrs[6][0:2]

	#print(real_csv_dir)
	write_to_csv_for_time_series(precip_real, real_csv_dir, "pr")
	#write_to_csv_for_time_series(tas_real, real_csv_dir, "tas")
	#write_to_csv_for_time_series(tasmin_real, real_csv_dir, "tasmin")
	#write_to_csv_for_time_series(tasmax_real, real_csv_dir, "tasmax")
	#write_to_csv_for_time_series(rhs_real, real_csv_dir, "rhs")
	#write_to_csv_for_time_series(rhsmin_real, real_csv_dir, "rhsmin")
	#write_to_csv_for_time_series(rhsmax_real, real_csv_dir, "rhsmax")

main()
