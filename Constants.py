
scenarios = ["rcp26", "rcp45", "rcp60", "rcp85", "historical"]
realizations = ["r1i1p1", "r2i1p1", "r3i1p1", "r4i1p1", "r5i1p1"]
#[lat, lon]

grid_cells_for_stat = {
	"ocean_high" : [118,19],#Barents sea
	"ocean_mid" : [105,34],#Labrador sea
	"ocean_low" : [71, 67],#Andaman Sea
	"land_high" : [108,111],#Alaska
	"land_mid" : [92, 55],#College park
	"land_low" : [18, 78] #Sahara
}



rhs_day_dir = 'rhs_day_MIROC5_'
rhs_min_dir = 'rhsmin_day_MIROC5_'
rhs_max_dir = 'rhsmax_day_MIROC5_'
tas_day_dir = 'tas_day_MIROC5_'
tas_max_dir = 'tasmax_day_MIROC5_'
tas_min_dir = 'tasmin_day_MIROC5_'
pr_day_dir = 'pr_day_MIROC5_'


clmt_vars = {
	'pr' : [pr_day_dir,'log_norm'],
	'tas' : [tas_day_dir,'stand'],
	'tasmin' : [tas_min_dir,'stand'],
	'tasmax' : [tas_max_dir,'stand'],
	'rhsmin' : [rhs_min_dir,'stand'],
	'rhsmax' : [rhs_max_dir,'stand'],
	'rhs' : [rhs_day_dir,'stand']
}


GB_to_B = 1073741824
