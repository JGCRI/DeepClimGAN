
from Constants import clmt_vars, grid_cells_for_stat
from scipy import stats
import sys
from numpy import np


N_DAYS = 0
N_CELLS = len(grid_cells_for_stat)

	
def merge_tensors(data_path):

	clmt_var_keys = list(clmt_vars.keys())

	tas_tsrs = tasmin_tsrs = tasmax_tsrs = []
	rhs_tsrs = rhsmin_tsrs = rhsmax_tsrs = []
	pr_tsrs = []

	#tensor dim: 7 x 128 x 256 x 32
        for filename in os.listdir(data_path):
                ts_name = os.path.join(data_path, filename)
                tsr = torch.load(ts_name)

		pr_tsrs.append(tsr[clmt_var_keys.index('pr')])
                tas_tsrs.append(tsr[clmt_var_keys.index('tas')])
                tasmin_tsrs.append(tsr[clmt_var_keys.index('tasmin')])
                tasmax_tsrs.append(tsr[clmt_var_keys.index('tasmax')])
                rhsmin_tsrs.append(tsr[clmt_var_keys.index('rhsmin')])
                rhsmax_tsrs.append(tsr[clmt_var_keys.index('rhsmax')])
                rhs_tsrs.append(tsr[clmt_var_keys.index('rhs')])

	pr_tsr = torch.cat(pr_tsrs, dim=3)
	tas_tsr = torch.cat(tas_tsrs, dim=3)
	tasmin_tsr = torch.cat(tasmin_tsrs, dim=3)
	tasmax_tsr = torch.cat(tasmax_tsrs, dim=3)
	rhs_tsr = torch.cat(rhs_tsrs, dim=3)
	rhsmin_tsr = torch.cat(rhsmin_tsrs, dim=3)
	rhsmax_tsr = torch.cat(rhsmax_tsrs, dim=3)
	N_DAYS = pr_tsr.shape[-1]
	
	return pr_tsr, tas_tsr, tasmin_tsr, tasmax_tsr, rhs_tsr, rhsmin_tsr, rhsmax_tsr

def get_tas_stat(tas_tsr):
	"""
	Output:
	 mean
	 sd/variance
	 p-value for Shapiro-Wilk
	"""
	
	dic = [[0 for _ in range(3)] for _ in range(N_CELLS)]
	
	for i, (cell, val) in enumerate(grid_cells_for_stat.items()):
		lat, lon = val
		tsr = tas_tsr[lat, lon]
		mean = torch.mean(tsr)
		std = torch.sqrt((tsr - mean) ** 2 / N_DAYS)
		test_stat, p_value = stats.shapiro(tsr)
		dic[i][:] = [mean, std, p_value]
	
	return dic	


def get_p_stat_for_cells(p_tsr, n_days):
	
	"return 5 params per each cell grid"
	
	dic = [[0 for _ in range(5)] for _ in range(N_CELLS)]
	n_months = pr_tsr.shape[0]
	n_days = n_months * 32
	
	for i, (cell, val) in enumerate(grid_cells_for_stat.items()):
		lat, lon = val
		tsr = pr_tsr[lat, lon]
		dic[i][:] = get_p_stat(tsr, n_days)
	return dic
						

def get_p_stat(pr, n_days):
	pr_min = np.min(pr)
	n_zero_vals = np.sum(pr == 0)
	zero_fraq = n_zero_values / n_days
	non_zero = []
	for day_p in pr:
		non_zero.append(day_p) if day_p != 0 else continue
	non_zero = np.asarray(non_zero)
	non_zero_mean = np.mean(non_zero)
	non_zero_std = np.std(non_zero)

	return [pr_min, zero_fraq, non_zero_mean, non_zero_std]


def get_tas_min_max_stat_for_cells(tas, tasmin, tasmax):

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
	
	tasmin_mean = np.mean(tasmin)
	tasmax_mean = np.mean(tasmax)
	
	tasmin_std = np.std(tasmin)
	tasmax_std = np.std(tasmax)
	
	tas_arr = np.asarray(tas)
	tas_filter = tas_arr[tas_arr >= tasmin and tas_arr <= tasmax]
	tas_is_in_range_fraq = np.sum(tas_filter, where=[True]) / tas.shape[0]
	
	return [tasmin_mean, tasmax_mean, tasmin_std, tasmax_std, tas_is_in_range_fraq]


def get_rhs_stat_for_cells(rhs):
	
	dic = [[0 for _ in range(4)] for _ in range(N_CELLS)]

	for i, (cell, val) in enumerate(grid_cells_for_stat.items()):
		lat, lon = val
		rhs_tsr = rhs[lat, lon]
		dic[i][:] = get_rhs_stat(rhs_tsr)

	return dic


def get_rhs_stat(rhs):
	rhs_min = np.min(rhs)
	rhs_max = np.max(rhs)
	rhs_mean = np.mean(rhs)
	rhs_std = np.std(rhs)	
	return (rhs_min, rhs_max, rhs_mean, rhs_std)
	
	
def get_rhs_min_max_stat_for_cells(rhs, rhsmin, rhsmax):
	
	dic = [[0 for _ in range(5)]for _ in range(N_CELLS)]
	
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
	rhsmin_mean = np.mean(rhsmin)
	rhsmax_mean = np.mean(rhsmax)
	
	rhsmin_std = np.std(rhsmin)
	rhsmax_std = np.std(rhsmax)

	rhs_arr = np.asarray(rhs)
	rhs_filter = rhs_arr[rhs_arr >= rhsmin and rhs_arr <= rhsmax]
	rhs_is_in_range_fraq = np.sum(rhs_filter, where=[True]) / rhs.shape[0]
	
	return [rhsmin_mean, rhsmax_mean, rhsmin_std, rhsmax_std, rhs_is_in_range_fraq]


def get_dewpoint_stat_for_cells(rhs, tas):
	
	dic = [[0 for _ in range(5)] for _ in range(N_CELLS)]

	for i, (cell, val) in enumerate(grid_cells_for_stat.items()):
		lat, lon = val
		rhs_tsr = rhs[lat, lon]
		tas_tsr = tas[lat, lon]
		dic[i][:] = get_dewpoint_stat(rhs_tsr, tas_tsr)

	return dic


def calc_dewpoint(rhs, tas):
	lambda = 243.12 + 273
	alpha = 6.112
	beta = 17.62
	
	num = np.log(rhs / 100) + (beta * tas) / (lamda + tas)
	denom = np.log(rhs / 100) + (beta * tas) / (lambda + tas)

	dp = lambda * num / (beta - denom)
	
	return dp


def get_dewpoint_stat(rhs, tas):
	dp = calc_dewpoint(rhs, tas)
	dp_min = np.min(dp)
	dp_max = np.max(dp)
	dp_mean = np.mean(dp)
	dp_std = np.std(dp)

	return [dp, dp_min, dp_max, dp_mean, dp_std]




def main():
	
	gen_data_path = sys.argv[1]
	real_data_path = sys.argv[2]
	n_days = int(sys.argv[3]) * 32
	
	pr_gen, tas_gen, tasmin_gen, tasmax_gen, rhs_gen, rhsmin_gen, rhsmax_gen = merge_tensors(gen_data_path)
	#pr_real, tas_real, tasmin_real, tasmax_real, rhs_real, rhsmin_real, rhsmax_real = merge_tensors(real_data_path)
	

	tas_stats = get_tas_stat()
	print(tas_stats)
	
