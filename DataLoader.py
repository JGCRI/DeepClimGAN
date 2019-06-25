import os
import netCDF4 as n
import numpy as np
import torch.nn as nn
import torch
from Constants import clmt_vars

#default params
lat = 128
lon = 256
time = 30
data_dir = '../clmt_data/'
n_channels = len(clmt_vars)



def export_netcdf(filename, var_name):
	print(filename)
	nc = n.Dataset(filename, 'r', format='NETCDF4_CLASSIC')
	print(nc.variables.keys())
	lon = nc.variables['lon'][:]
	lat = nc.variables['lat'][:]
	var = nc.variables[var_name][:]
	
	return var




def main():
	
	for key, val in clmt_vars.items():
		file_dir = data_dir + val[1] + '/'
		#create tensor for one climate variable
		concat_tensor = None
		#sort files in ascending order (based on the date)
		filenames = os.listdir(file_dir)
		filenames = sorted(filenames)
		
		for i, filename in enumerate(filenames):
			print(filename, i)
			raw_clmt_data = export_netcdf(file_dir + filename,key)
			lon, lat = raw_clmt_data.shape[1], raw_clmt_data[2]	 
			raw_tensor = torch.tensor(raw_clmt_data, dtype=torch.float32)
			print(raw_tensor.shape)
			if i == 0: 
				concat_tensor = raw_tensor
			else:	
				#concatenate tensors along the size dimension
				concat_tensor = torch.cat((concat_tensor, raw_tensor), 0)
		print(concat_tensor.shape)
	



main()
