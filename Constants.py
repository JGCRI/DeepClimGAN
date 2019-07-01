
huss_dir = 'huss_day/'
rhs_day_dir = 'rhs_day/'
rhs_min_dir = 'rhsmin_day/'
rhs_max_dir = 'rhsmax_day/'
tas_day_dir = 'tas_day/'
tas_max_dir = 'tasmax_day/'
tas_min_dir = 'tasmin_day/'
pr_day_dir = 'pr_day/'


#value: [dir_name, normalization_type]
clmt_vars = {
	'pr' : [pr_day_dir,'stand'],
	'tas' : [tas_day_dir,'norm'],
#	'tasmin' : [tas_min_dir,'norm'],
#	'tasmax' : [tas_max_dir,'norm'],
#	'rhsmin' : [rhs_min_dir,''],
#	'rhsmax' : [rhs_max_dir,''],
#	'rhs' : [rhs_day_dir,'norm'],
#	'huss' : [huss_dir,'']


}
