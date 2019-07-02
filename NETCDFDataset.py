import numpy as np
import torch
import os
import netCDF4 as n
import torch.nn as nn
import random
from Constants import clmt_vars
from torch.utils import data

#default params
lat = 128
lon = 256
time = 30
n_channels = len(clmt_vars)
#number of days to look back and in the future
context_window = 5

data_dir = '../clmt_data/'

class NETCDFDataset(data.Dataset):

	def __init__(self, data_dir, train_pct):

		self.len, self.file_to_idx = self.count_days_in_file(data_dir)
		self.train_len, self.dev_len, self.test_len = self.create_split_bounds(self.len, train_pct)				
#		self.data = self.build_dataset(data_dir)


	def __len__(self):
		"""
		Number of months (datapoints ) in the dataset
		"""
		return self.len

	def __getitem__(self, idx):
		"""
		Extracts one datapoint, which is one month of records.
		Combines all the context needed for network
		param: idx (int) batch idx 

			
		return:
		input - final input to the network (H x W x 40 x N_channels + 2)
		current_month - one month of records to compute the loss (H x W x 30 x N_channels)
		
		"""
	
		#first 5 days are reserved for the context
		start = 5
		n_days = 30
		current_month = self.data[:, :, (start + idx):(start + n_days + idx), :]#output size is H x W x 30 x N_channels
		#TODO: handle the case where prev or next is in different file
		prev_5 = self.data[:, :, (start + idx - 5):(start + idx), :]
		next_5 = self.data[:, :, (start + idx):(start + idx + 5), :]
		###
		
		
		keys = list(clmt_vars.keys())
		pr_idx = keys.index('pr')
		tas_idx = keys.index('tas')
		avg_p =	self.expand_map(current_month[:,:,:,pr_idx])
		avg_t =	self.expand_map(current_month[:,:,:,tas_idx])
		avg_contexts = torch.cat([avg_p, avg_t], dim=3)
		last_current_next_month= torch.cat([current_month, prev_5, next_5], dim=2)
		input = torch.cat([last_current_next_month, avg_contexts], dim=3)#input to the network together with a context

		return input, current_month



	def export_netcdf(self,filename, var_name):
                """
                Export data from one .nc file

                :param filename: (str) Name of the .nc file
                :param var_name: (str) Name of the climate variable

                :return exported from .nc file variable
                """

                nc = n.Dataset(filename, 'r', format='NETCDF4_CLASSIC')
                lon = nc.variables['lon'][:]
                lat = nc.variables['lat'][:]
                var = nc.variables[var_name][:]
                return var
	
	def count_days_in_dataset(self, data_dir):
		"""We can pass name of the file of any variable
		  since we assume that number of days per each variable is the same.
		  We will also store the range of indexes relatively to the whole dataset,
		  and their mapping to the name of the file.		
		"""
		clmt_var = clmt_vars.keys[0]
		file_dir = data_dir + clmt_dir
		filenames = os.listdir(file_dir)
		filenames = sorted(filenames)
		#mapping from indexes to filenames
		file_to_idx = {}
		dataset_len = 0
		#for i, filename in enumerate(filenames):
			nc = n.Dataset(filename, 'r', format='NETCDF4_CLASSIC')	
			data_len = nc.variables[var_name][:].shape[0]
			start, end = dataset_len, dataset_len + data_len - 1
			file_to_idx[filename] = [start, end]
			dataset_len += data_len

		return dataset_len, file_to_idx
	

	def build_dataset(self, data_dir):
                """
                Builds dataset out of all climate variables of shape HxWxNxC, where:
                        N - #of datapoints (days)
                        H - latitude
                        W - longitude
                        C - #of channels (climate variables)

                :return dataset as a tensor
                """

                #all tensors is a list of tensor, where each tensor has data about one climate variable
                all_tensors = []
                count_files = 0
                for i, (key, val) in enumerate(clmt_vars.items()):
                        clmt_dir = val[0]
                        file_dir = data_dir + clmt_dir
                        #create tensor for one climate variable
                        tensors_per_clmt_var = []
                        #sort files in ascending order (based on the date)
                        filenames = os.listdir(file_dir)
                        filenames = sorted(filenames)
                        count_files += len(filenames)
                        for j, filename in enumerate(filenames):
                                raw_clmt_data = self.export_netcdf(file_dir + filename,key)
                                raw_tsr= torch.tensor(raw_clmt_data, dtype=torch.float32)
                                tensors_per_clmt_var.append(raw_tsr)

                        #concatenate tensors along the size dimension
                        concat_tsr = torch.cat(tensors_per_clmt_var, dim=0)
                        all_tensors.append(concat_tsr)

                        print("Finished parsing {} files for variable \"{}\" ".format(len(filenames),key))
                res_tsr = torch.stack(all_tensors, dim=3)
                print("Finished parsing {} files total".format(count_files))
		
                #permuate tensor for convenience
                res_tsr = res_tsr.permute(1, 2, 0, 3)#H x W x N x N
                print("The size of result tensor is {}".format(res_tsr.shape))
                return res_tsr


	def create_split_bounds(self, N, train_pct):
                #reference: https://github.com/hutchresearch/ml_climate17/blob/master/resnet/utils/DataHelper.py
                """
                Computes split bounds for train, dev and test.

                :param N (int) Length of the dataset
                :param train_pct=0.7 (float) Percent of data to use for training

                :return: train (default to 70% of all data)
                :return: dev (default to 15% of all data)
                :return: test (default to 15% of all data)
                """


                train_len = int(round(train_pct * N))
                if (N - train_len) % 2 == 1:
                        train_len += 1

                #NOTE: We assume that dev_len = test_len
                dev_len = test_len = int((N - train_len) / 2)

                assert "Not all data points are being used. Check create_split_bounds()", \
                        (train_len + dev_len + test_len) == N

		
                return train_len, dev_len, test_len
	


	def get_noise(self, N):
                """
                Creates a multivariate normal (Gaussian) distribution
                parametrized by a mean vector and a covariance matrix

                param: N (int) Dimension of a distribution

                return sample from N-dimensional Gaussian distribution
                """
                m = MultivariateNormal(torch.zeros(N), torch.eye(N))
                m.sample()
                return m
	
	def expand_map(self, x):
		map = x.mean(2)#H x W x 1 x 1, we want to expand it to H x W x 40 x 1
		map = map.unsqueeze(-1).unsqueeze(-1)
		expanded_map = torch.cat(list(torch.split(map, 1, dim=2)) * 40, dim=2)
		return expanded_map
			
		




def main():
	
	import time
	train_pct = 0.7
	batch_size = 128
	start = time.time()
	ds = NETCDFDataset(data_dir, train_pct)
	dl = data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
	for batch_idx, batch in enumerate(dl):
		input, current_month = batch
		print(input.shape, current_month.shape)
	print(time.time() - start)


	
main()
	
