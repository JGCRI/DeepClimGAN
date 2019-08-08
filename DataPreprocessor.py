import numpy as np
import torch
import os
import netCDF4 as n
from Constants import clmt_vars, scenarios, realizations
import sys
from Normalizer import Normalizer
import torch
import torch.distributed as dist
import argparse


class DataPreprocessor():

	def __init__(self, data_dir, scenarios, realizations, start_date, end_date):
		self.total_len, self.n_files = self.build_save_tensors_for_all_scenarios(data_dir, scenarios, realizations, start_date, end_date)
	
	def build_save_tensors_for_all_scenarios(self, data_dir,scenarios, realizations, start_date, end_date):
		total_len = 0
		save_dir = sys.argv[2]
		filenames = os.listdir(data_dir)
		filenames = sorted(filenames)
		n_files = 0
		for scenario in scenarios:
			for realiz in realizations:
				prefix = scenario + "_" + realiz
				tensor, tensor_len, num_files = self.build_dataset(data_dir, filenames, prefix, start_date, end_date)
				#save tensor as torch array to the folder
				ts_name = os.path.join(save_dir, prefix)
				torch.save(tensor, ts_name + ".pt")
				total_len += tensor_len
				n_files += num_files
		
		return total_len, n_files
	

	def export_netcdf(self,filename, var_name):
                """
                Export data from one .nc file
                :param filename: (str) Name of the .nc file
                :param var_name: (str) Name of the climate variable
                :return exported from .nc file variable
                """

                nc = n.Dataset(filename, 'r', format='NETCDF4_CLASSIC')
                var = nc.variables[var_name][:]
                return var


	def build_dataset(self, data_dir, filenames, prefix, start_date, end_date):
		"""
		Builds dataset out of all climate variables of shape HxWxNxC, where:
		N - #of datapoints (days)
		H - latitude
		W - longitude
		C - #of channels (climate variables)
		param: data_dir(str) directory with the data
		param: data_pct (float) percent of the data to use
		:return dataset as a tensor
		"""
		n_files = 0
		all_tensors = []
		for i, (key, val) in enumerate(clmt_vars.items()):
			file_dir = data_dir
			var_name = val[0]
			#create tensor for one climate variable
			tensors_per_clmt_var = []
		
			#search all that suit to our scenario
			my_files = []
			pattern = var_name + prefix
			for i in range(len(filenames)):
				file_end_year = int(filenames[i][-11:-7])
				file_start_year = int(filenames[i][-20:-16])
				if pattern in filenames[i] and file_end_year <= end_date and file_start_year >= start_date:
					my_files.append(filenames[i])
		
			n_files += len(my_files)
			for j, filename in enumerate(my_files):
				raw_clmt_data = self.export_netcdf(file_dir + filename, key)
				raw_tsr = torch.tensor(raw_clmt_data, dtype=torch.float32)
				tensors_per_clmt_var.append(raw_tsr)
		
			#concatenate tensors along the size dimension
			concat_tsr = torch.cat(tensors_per_clmt_var, dim=0)
			all_tensors.append(concat_tsr)


		for t in all_tensors:
			print(t.shape)		
		res_tsr = torch.stack(all_tensors, dim=3)
		tensor_len = res_tsr.shape[0]
		
		res_tsr = res_tsr.permute(3, 1, 2, 0)#C x H x W x N
		return res_tsr, tensor_len, n_files
		
		
def main():
	
	file_dir = sys.argv[1]
	start_date = int(sys.argv[3])
	end_date = int(sys.argv[4])
	#load and save files as tensors
	processor = DataPreprocessor(file_dir, scenarios, realizations, start_date, end_date)	

		
main()
