
rhs_day_dir = 'rhs_day_MIROC5_rcp26_r1i1p1_20060101-20091231.nc'
rhs_min_dir = 'rhsmin_day_MIROC5_rcp26_r1i1p1_20060101-20091231.nc'
rhs_max_dir = 'rhsmax_day_MIROC5_rcp26_r1i1p1_20060101-20091231.nc'
tas_day_dir = 'tas_day_MIROC5_rcp26_r1i1p1_20060101-20091231.nc'
tas_max_dir = 'tasmax_day_MIROC5_rcp26_r1i1p1_20060101-20091231.nc'
tas_min_dir = 'tasmin_day_MIROC5_rcp26_r1i1p1_20060101-20091231.nc'
pr_day_dir = 'pr_day_MIROC5_rcp26_r1i1p1_20060101-20091231.nc'


#value: [dir_name, normalization_type]
clmt_vars = {
	'pr' : [pr_day_dir,'log_norm'],
	'tas' : [tas_day_dir,'stand'],
	'tasmin' : [tas_min_dir,'stand'],
	'tasmax' : [tas_max_dir,'stand'],
	'rhsmin' : [rhs_min_dir,'stand'],
	'rhsmax' : [rhs_max_dir,'stand'],
	'rhs' : [rhs_day_dir,'stand']
}
