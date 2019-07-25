
rhs_day_dir = 'rhs_day_MIROC5'
rhs_min_dir = 'rhsmin_day_MIROC5'
rhs_max_dir = 'rhsmax_day_MIROC5'
tas_day_dir = 'tas_day_MIROC5'
tas_max_dir = 'tasmax_day_MIROC5'
tas_min_dir = 'tasmin_day_MIROC5'
pr_day_dir = 'pr_day_MIROC5'


clmt_vars = {
	'pr' : [pr_day_dir,'log_norm'],
	'tas' : [tas_day_dir,'stand'],
	'tasmin' : [tas_min_dir,'stand'],
	'tasmax' : [tas_max_dir,'stand'],
	'rhsmin' : [rhs_min_dir,'stand'],
	'rhsmax' : [rhs_max_dir,'stand'],
	'rhs' : [rhs_day_dir,'stand']
}
